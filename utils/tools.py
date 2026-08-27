from __future__ import annotations

import os
import gc
import json
import math
import subprocess
import cv2
import torch
import numpy as np

from collections import OrderedDict
from pathlib import Path
from threading import Lock
from dotenv import load_dotenv
from langchain.tools import tool
from decord import VideoReader, cpu
from typing import Generator, Optional

from utils.env import env_first, env_float, env_get, env_int
from utils.video_analyzer_model import Video_Analyzer, pick_device
from utils.video_process import generate_frames
from utils.vlm import VLM_MAX_FRAMES, VLM_Manager
from utils.video_export import VideoExportError, export_video
from utils.clip_archive import ArchiveCategory, ArchiveError, archive_clip

load_dotenv()

AS_MODEL_NAME = env_first("AS_MODEL_NAME", "ANALYZER_BACKBONE") or "s3d"
AS_OVERLAP = env_int("AS_OVERLAP", 8)
AS_FC_CHECKPOINT = env_first("AS_FC_CHECKPOINT", "ANALYZER_FC_CHECKPOINT") or None
AS_TRT_PLAN_PATH = env_get("AS_TRT_PLAN_PATH") or None
AS_BATCH = env_int("AS_BATCH", 4)
EVREN_API_KEY = env_get("EVREN_API_KEY") or None
EVREN_URL = env_first("EVREN_URL", "EVREN_BASE_URL") or None
VLM_SYSTEM_PROMT = env_get("VLM_SYSTEM_PROMT") or None
AS_CLIP_SIZE = env_int("AS_CLIP_SIZE", 16)
AS_FPS = env_int("AS_FPS", 30)
AS_WIDTH = env_int("AS_WIDTH", 224)
AS_HEIGHT = env_int("AS_HEIGHT", 224)
AS_STRIDE = env_int("AS_STRIDE", AS_CLIP_SIZE - AS_OVERLAP)
VLM_SOURCE_SAMPLE_FPS = env_float("VLM_SOURCE_SAMPLE_FPS", 5.0)

DEVICE = pick_device()
_ROOT = Path(__file__).resolve().parents[1]
_anomaly_segment_model: Optional[Video_Analyzer] = None


class ToolInputError(ValueError):
    def __init__(self, code: str, message: str, data: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.data = data or {}


def _tool_result(
    *,
    ok: bool,
    data: Optional[dict] = None,
    warnings: Optional[list[dict]] = None,
    error: Optional[dict] = None,
) -> str:
    """Bütün agent tool sonuçlarını tek JSON sözleşmesine dönüştürür."""
    return json.dumps(
        {
            "ok": ok,
            "data": data or {},
            "warnings": warnings or [],
            "error": error,
        },
        ensure_ascii=False,
    )


def _tool_error(code: str, message: str, data: Optional[dict] = None) -> str:
    return _tool_result(
        ok=False,
        data=data,
        error={"code": code, "message": message},
    )


def _resolve_checkpoint() -> Optional[str]:
    for candidate in (
        AS_FC_CHECKPOINT,
        env_get("ANALYZER_FC_CHECKPOINT"),
        "Checkpoint/best_loss_fold_3.pt",
    ):
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = _ROOT / path
        if path.is_file():
            return str(path)
    return AS_FC_CHECKPOINT


def get_anomaly_segment_model() -> Video_Analyzer:
    """Analyzer'ı ilk tool çağrısında yükle. TRT yalnız CUDA'da."""
    global _anomaly_segment_model
    if _anomaly_segment_model is not None:
        return _anomaly_segment_model

    ckpt = _resolve_checkpoint()
    if ckpt and not Path(ckpt).is_file():
        raise FileNotFoundError(f"FC checkpoint yok: {ckpt}")

    print(f"[info] Analyzer yükleniyor device={DEVICE} ckpt={ckpt}")
    model = Video_Analyzer(
        AS_MODEL_NAME,
        AS_CLIP_SIZE,
        AS_OVERLAP,
        ckpt,
    ).eval()
    model.to(DEVICE)
    if DEVICE.type == "cuda" and AS_TRT_PLAN_PATH:
        model.export_trt(AS_TRT_PLAN_PATH, AS_BATCH, (AS_HEIGHT, AS_WIDTH))
    else:
        print(f"[info] Analyzer {DEVICE}; TensorRT atlandı.")
    _anomaly_segment_model = model
    return model



VIDEO_TOO_SHORT_ERROR = lambda vp, ms, vs : f"Belirtilen {vp} dosyası video analiz segmentasyonu gerçekleştirebilemk için çok kısa! Lütfen {ms} saniyeden daha fazla olan video ile deneyin. Sizin videonuz {vs} saniye yalnızca."

_METADATA_CACHE_LIMIT = 128
_metadata_cache: OrderedDict[str, tuple[tuple, dict]] = OrderedDict()
_metadata_cache_lock = Lock()
# Sabit sayıda kilit: video sayısıyla büyümez; aynı dosyanın ilk okumasını korur.
_metadata_read_locks = tuple(Lock() for _ in range(32))


def _video_file_signature(video_path: str) -> tuple:
    stat = os.stat(video_path)
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _get_video_metadata(video_path: str) -> dict:
    """Yalnız metadata'yı önbellekler; canlı Decord okuyucusu paylaşılmaz."""
    path = os.path.realpath(video_path)
    with _metadata_read_locks[hash(path) % len(_metadata_read_locks)]:
        signature = _video_file_signature(path)
        with _metadata_cache_lock:
            cached = _metadata_cache.get(path)
            if cached is not None:
                if cached[0] == signature:
                    _metadata_cache.move_to_end(path)
                    return dict(cached[1])
                del _metadata_cache[path]

        vr = VideoReader(path, ctx=cpu(0))
        try:
            total_frames = len(vr)
            fps = float(vr.get_avg_fps() or 0.0)
            if total_frames <= 0 or not math.isfinite(fps) or fps <= 0:
                raise ToolInputError(
                    "INVALID_VIDEO",
                    "Video süresi hesaplanamadı: geçerli frame veya FPS bilgisi yok.",
                )
            shape = vr[0].shape
            metadata = {
                "duration_sec": total_frames / fps,
                "fps": fps,
                "frame_count": total_frames,
                "width": int(shape[1]),
                "height": int(shape[0]),
            }
        finally:
            del vr

        if _video_file_signature(path) != signature:
            raise ToolInputError("VIDEO_CHANGED", "Video okunurken dosya değişti; isteği tekrar deneyin.")
        with _metadata_cache_lock:
            _metadata_cache[path] = (signature, metadata)
            _metadata_cache.move_to_end(path)
            while len(_metadata_cache) > _METADATA_CACHE_LIMIT:
                _metadata_cache.popitem(last=False)
        return dict(metadata)


def _validate_video_range(video_path: str, start_sec: float, end_sec: float) -> tuple[float, float, dict, bool]:
    """Aralığı doğrular; yalnızca video sonunu aşan end_sec değerini kırpar."""
    start = float(start_sec)
    end = float(end_sec)
    if not math.isfinite(start) or not math.isfinite(end):
        raise ToolInputError("INVALID_TIME_RANGE", "start_sec ve end_sec sonlu sayılar olmalıdır.")
    if start < 0:
        raise ToolInputError("INVALID_TIME_RANGE", "start_sec negatif olamaz.")
    if end <= start:
        raise ToolInputError(
            "INVALID_TIME_RANGE",
            "end_sec, start_sec değerinden büyük olmalıdır.",
        )

    metadata = _get_video_metadata(video_path)
    duration = float(metadata["duration_sec"])
    if start >= duration:
        raise ToolInputError(
            "TIME_OUT_OF_RANGE",
            f"İstenen başlangıç ({start:.2f}sn) video süresinin "
            f"({duration:.2f}sn) dışındadır; bu zaman analiz edilemez.",
            data={"video": metadata, "requested_range": {"start_sec": start, "end_sec": end}},
        )

    effective_end = min(end, duration)
    return start, effective_end, metadata, effective_end != end

def get_frame_id(video_path: str, number: int, unit: str) -> int:

    unit_map= {
        "saat": 3600,
        "dakika": 60,
        "saniye": 1
    }

    if not unit in unit_map:
        print("Kullanılan birim anlaşılamadı.")

    second = unit_map[unit]

    seconds = int(number * second)

    vr = VideoReader(video_path)
    fps = vr.get_avg_fps()

    frame_id = int(round(seconds * fps))

    return frame_id

def create_clip_generator(
        video_path: os.PathLike,
        clip_size: int,
        stride: int,
        fps: int,
        width: int,
        height: int) -> tuple[Generator[np.array, None, None], int]|str:

    vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height, num_threads=2)
    org_fps = vr.get_avg_fps() or 1.0
    frame_indices = torch.arange(0, len(vr), step=org_fps / fps).long()
    total_frames = len(frame_indices)

    video_seconds = len(vr) / org_fps
    minimum_seconds = clip_size / fps

    if video_seconds < minimum_seconds:
        return VIDEO_TOO_SHORT_ERROR(video_path, minimum_seconds, video_seconds)

    number_of_clips = 1 + (total_frames - clip_size) // stride

    def _generator(vr:VideoReader):
        try:
            for end_idx in range(clip_size, total_frames + 1, stride):
                start_idx = end_idx - clip_size

                clip_indices = frame_indices[start_idx:end_idx]
                clip = vr.get_batch(clip_indices.tolist()).asnumpy()
                yield torch.from_numpy(clip)

        finally:
            del vr
            gc.collect()

    return _generator(vr), number_of_clips


@tool
def run_abnormal_event_segmenter(video_path: str) -> str:
    """Videonun tamamını anomali segmentasyon modeliyle tarar.

    Eşik üstündeki zaman aralıklarını ve skorlarını, kullanılan eşik ile video
    metadata bilgisiyle birlikte döndürür. Olay türünü kesin sınıflandırmaz;
    boş segment listesi yalnızca mevcut eşikte anomali bulunmadığını gösterir.
    Sonuç ortak tool JSON zarfındadır.
    """
    if not video_path or not os.path.exists(video_path):
        return _tool_error("FILE_NOT_FOUND", f"Video bulunamadı: {video_path}")

    metadata = None
    try:
        metadata = _get_video_metadata(video_path)
        model = get_anomaly_segment_model()
        save_dir = str(_ROOT / "_stuff" / "lab_runs")
        segments = model.analyze(
            video_path=video_path,
            width=AS_WIDTH,
            height=AS_HEIGHT,
            fps=AS_FPS,
            batch_size=AS_BATCH,
            threshold=0.3,
            save_graph=True,
            save_clips=False,
            save_dir=save_dir,
        )
        warnings = []
        if not segments:
            warnings.append({
                "code": "NO_SEGMENTS_ABOVE_THRESHOLD",
                "message": "Mevcut eşikte anormal segment bulunmadı; bu sonuç videonun kesin olarak normal olduğunu kanıtlamaz.",
            })
        return _tool_result(
            ok=True,
            data={
                "video_path": video_path,
                "video": metadata,
                "analysis_scope": "full_video",
                "threshold": 0.3,
                "segment_count": len(segments),
                "segments": segments,
            },
            warnings=warnings,
        )
    except ToolInputError as e:
        return _tool_error(e.code, str(e), data=e.data)
    except FileNotFoundError as e:
        return _tool_error("FILE_NOT_FOUND", str(e))
    except Exception as e:
        print(f"[segmenter hata] {type(e).__name__}: {e}")
        return _tool_error(
            "SEGMENTER_ERROR",
            f"{type(e).__name__}: {e}",
            data={"video_path": video_path, "video": metadata} if metadata else {"video_path": video_path},
        )

def parse_time_to_frame_id(number:int, unit:str, len_frames, fps) -> int:
    unit_map = {
        "saat":3600,
        "dakika":60,
        "saniye":1
    }
    if not unit in unit_map.keys():
        return f"Kullanılan birim anlışlamadı: {unit}! Birim dönüşümünde kullanılabien birimler: {unit_map.keys()}."

    second = unit_map[unit]
    

@tool
def analyze_video_with_vlm(video_path: str, query: str, start_sec: float, end_sec: float) -> str:
    """Videonun belirtilen zaman aralığını görsel-dil modeliyle inceler.

    Görsel olay açıklama, sayma, karşılaştırma ve görüntüdeki metni okuma gibi
    soruları yanıtlar. Başlangıç video dışında ise çağrı reddedilir; yalnız bitiş
    taşıyorsa video sonuna kırpılır ve uyarı döner. Sonuç ortak tool JSON zarfındadır.
    """
    try:
        valid_start, valid_end, metadata, was_clamped = _validate_video_range(
            video_path, start_sec, end_sec
        )
        vlm = VLM_Manager(
            api_key=EVREN_API_KEY,
            base_url=EVREN_URL,
            system_prompt=VLM_SYSTEM_PROMT,
        )
        frames, actual_start, actual_end = generate_frames(
            video_path=video_path,
            start_sec=valid_start,
            end_sec=valid_end,
            all_video=False,
            FPS=VLM_SOURCE_SAMPLE_FPS,
            max_frames=VLM_MAX_FRAMES
        )
        source_time_context = (
            "İncelenen geçici klip, kaynak videonun "
            f"{actual_start:.2f}–{actual_end:.2f} saniye aralığından örneklenmiştir. "
            f"Geçici klipte 0:00, kaynak videoda {actual_start:.2f}. saniyeye karşılık gelir. "
            "Zaman belirten cevaplarda kaynak video zamanını kullan.\n\n"
            f"Kullanıcı sorusu: {query}"
        )
        # Kare zamanları FPS yuvarlaması nedeniyle birkaç salise kayabilir; geçici
        # MP4 süresini kullanıcının/segmenterin istediği gerçek aralığa sabitle.
        source_duration = valid_end - valid_start
        response = vlm.run(
            text=source_time_context,
            frames=frames,
            source_duration=source_duration,
        )
        
        del frames
        gc.collect()

        if "[VLM HATA]:" in response:
            return _tool_error(
                "VLM_SERVICE_ERROR",
                response.split("[VLM HATA]:", 1)[1].strip(),
                data={"video_path": video_path, "video": metadata},
            )

        warnings = []
        if was_clamped:
            warnings.append({
                "code": "END_TIME_CLAMPED",
                "message": "İstenen bitiş video süresini aştığı için aralık video sonunda sınırlandırıldı.",
                "requested_end_sec": float(end_sec),
                "effective_end_sec": valid_end,
            })
        return _tool_result(
            ok=True,
            data={
                "video_path": video_path,
                "video": metadata,
                "requested_range": {"start_sec": float(start_sec), "end_sec": float(end_sec)},
                "effective_range": {"start_sec": valid_start, "end_sec": valid_end},
                "sampled_range": {"start_sec": actual_start, "end_sec": actual_end},
                "vlm_response": response,
            },
            warnings=warnings,
        )

    except ToolInputError as e:
        return _tool_error(e.code, str(e), data=e.data)
    except FileNotFoundError as e:
        return _tool_error("FILE_NOT_FOUND", str(e))
    except Exception as e:
        return _tool_error("VLM_ERROR", f"{type(e).__name__}: {e}")

@tool
def save_video_segment(video_path: str, start_sec: float, end_sec: float, output_filename: str) -> str:
    """Kaynak videonun belirtilen zaman aralığını MP4 dosyası olarak kaydeder.

    Çıktı adında uzantı yoksa `.mp4` ekler. Başlangıç video dışında ise çağrı
    reddedilir; yalnız bitiş taşıyorsa video sonuna kırpılır. FFmpeg çalışması ve
    oluşan videonun okunabilirliği doğrulanır. Sonuç ortak tool JSON zarfındadır.
    """
    if not video_path or not os.path.isfile(video_path):
        return _tool_error("FILE_NOT_FOUND", f"Kaynak video bulunamadı: {video_path}")
    if not (output_filename or "").strip():
        return _tool_error("INVALID_OUTPUT_PATH", "output_filename boş olamaz.")

    output_path = Path(output_filename.strip()).expanduser()
    if output_path.suffix.lower() != ".mp4":
        output_path = Path(f"{output_path}.mp4")
    if not output_path.parent.is_dir():
        return _tool_error("INVALID_OUTPUT_PATH", f"Çıktı klasörü bulunamadı: {output_path.parent}")

    try:
        valid_start, valid_end, metadata, was_clamped = _validate_video_range(
            video_path, start_sec, end_sec
        )
        export_video(video_path, output_path, valid_start, valid_end)

        warnings = []
        if was_clamped:
            warnings.append({
                "code": "END_TIME_CLAMPED",
                "message": "İstenen bitiş video süresini aştığı için video sonunda sınırlandırıldı.",
                "requested_end_sec": float(end_sec),
                "effective_end_sec": valid_end,
            })
        return _tool_result(
            ok=True,
            data={
                "video_path": video_path,
                "video": metadata,
                "requested_range": {"start_sec": float(start_sec), "end_sec": float(end_sec)},
                "saved_range": {"start_sec": valid_start, "end_sec": valid_end},
                "output_path": str(output_path),
                "output_size_bytes": output_path.stat().st_size,
            },
            warnings=warnings,
        )
    except VideoExportError as e:
        return _tool_error(e.code, str(e))
    except ToolInputError as e:
        return _tool_error(e.code, str(e), data=e.data)
    except FileNotFoundError:
        return _tool_error("FFMPEG_NOT_FOUND", "FFmpeg bulunamadı veya çalıştırılamadı.")
    except subprocess.TimeoutExpired:
        return _tool_error("FFMPEG_TIMEOUT", "FFmpeg 300 saniye içinde tamamlanamadı.")
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or "FFmpeg hata ayrıntısı döndürmedi.").strip()
        if len(detail) > 500:
            detail = detail[-500:]
        return _tool_error("FFMPEG_ERROR", detail)
    except Exception as e:
        return _tool_error("CLIP_EXPORT_ERROR", f"{type(e).__name__}: {e}")

@tool
def get_video_info(video_path: str) -> str:
    """Yerel video dosyasının teknik metadata bilgisini okur.

    Süre, FPS, toplam frame sayısı ve çözünürlük döndürür; görsel içeriği veya
    anomalileri analiz etmez. Sonuç ortak tool JSON zarfındadır.
    """
    try:
        info = _get_video_metadata(video_path)
        return _tool_result(ok=True, data={"video_path": video_path, "video": info})
    except FileNotFoundError as e:
        return _tool_error("FILE_NOT_FOUND", str(e))
    except ToolInputError as e:
        return _tool_error(e.code, str(e), data=e.data)
    except Exception as e:
        return _tool_error("VIDEO_INFO_ERROR", f"{type(e).__name__}: {e}")

@tool
def detect_and_track_objects(
    video_path: str,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    classes: Optional[list[str]] = None,
    render_video: bool = False,
) -> str:
    """Videoda nesneleri YOLO11 ile bulur; ByteTrack ile video içi takip yapar.

    Zamanlar kaynak videonun saniyeleridir. end_sec verilmezse video sonuna
    kadar inceler; başlangıç dışarıdaysa hata, yalnız bitiş taşıyorsa kırpma
    uyarısı döner. Her kare işlenir; uzun işler sınırı aşarsa daha kısa aralık
    istenir, sessizce kare atlanmaz. classes=None tüm COCO sınıflarıdır;
    filtre için İngilizce sınıf adları kullan: person, car, truck, bus, bicycle,
    motorcycle, backpack, handbag, suitcase gibi. Boş veya desteklenmeyen
    liste hata ve desteklenen sınıfları döndürür. Renk, kişi kimliği, silah,
    plaka metni veya suç türü tespiti yapmaz; olay yorumu için VLM gerekir.

    data.intervals içinde kind=class sınıfın, kind=track takip kimliğinin
    görüldüğü kesintisiz aralıklardır; end_sec hariçtir. Görünmeyen aralar
    birleştirilmez. Sınıfa göre klip isteniyorsa kind=class aralıklarını kullan.
    detection_count karelerdeki toplam kutu sayısıdır, tekil nesne sayısı değildir.
    Klip kaydetme için bu aralıklar save_video_segment'e
    verilebilir. Sınıf aralıkları öncelikli en çok 100 aralık özettedir;
    tamamı intervals_path dosyasında,
    kare kutuları/güvenleri/kaynak zamanları details_path dosyasındadır.
    Özet kesilmişse bütün sonuçları gördüğünü iddia etme; gerektiğinde daha
    dar zaman aralıklarıyla incele.
    track_id yalnız bu analize aittir; kaybolma/örtüşmede kimlik değişebilir.
    Tespit edilememesi nesnenin kesinlikle bulunmadığı anlamına gelmez.

    render_video=True nesneyi takip eden kutulu MP4 üretir (FFmpeg ve PyAV gerekir);
    kaynak dosyayı değiştirmez. Çıktı yolu annotated_video_path içindedir.
    Kaynak kare zamanları korunur. Aynı video/aralık/ayar için geçerli tespitler
    çizimde yeniden kullanılır: detection_cache_hit=True tekrar YOLO çalışmadığını,
    cache_hit=True istenen çıktının da hazırdan geldiğini belirtir.
    Sonuç ortak {ok, data, warnings, error} JSON zarfındadır.
    """
    from utils.object_tracking import TrackingError, track_objects

    try:
        if not video_path or not os.path.isfile(video_path):
            return _tool_error("FILE_NOT_FOUND", "Kaynak video bulunamadı.")
        requested_end = end_sec
        if end_sec is None:
            end_sec = _get_video_metadata(video_path)["duration_sec"]
        start, end, metadata, clamped = _validate_video_range(video_path, start_sec, end_sec)
        data, warnings = track_objects(video_path, start, end, classes, render_video)
        if clamped:
            warnings.append({"code": "END_TIME_CLAMPED", "message": "Bitiş video sonunda sınırlandırıldı."})
        return _tool_result(ok=True, data={
            **data, "video_path": video_path, "video": metadata,
            "requested_range": {"start_sec": start_sec, "end_sec": requested_end},
            "effective_range": {"start_sec": start, "end_sec": end},
        }, warnings=warnings)
    except (ToolInputError, TrackingError) as exc:
        return _tool_error(exc.code, str(exc), data=exc.data)
    except ImportError as exc:
        return _tool_error("OBJECT_DEPENDENCY_MISSING", f"Nesne tespiti bağımlılığı eksik: {exc}")
    except Exception as exc:
        return _tool_error("OBJECT_TRACKING_ERROR", f"{type(exc).__name__}: {exc}")


@tool
def detect_license_plate_regions(
    video_path: str,
    start_sec: float,
    end_sec: float,
) -> str:
    """Belirtilen video aralığında plaka adaylarını bulur ve PNG olarak kırpar.

    Kaynak video saniyeleri kullanılır; her kare incelenir, end_sec hariçtir.
    Tercihen ilgili kısa olay aralığını ver. Süre/kare/kırpım sınırı aşılırsa
    hata döner; sessizce eksik analiz yapılmaz. Bitiş video sonuna kırpılabilir.
    Plaka metnini OKUMAZ, araç/kişi kimliği veya takip yapmaz. Metin gerekirse
    bu tool'un details_path çıktısı read_license_plate_crops aracına verilebilir.
    Kırpımlar orijinal çözünürlükte plaka adaylarıdır; aynı plaka farklı
    karelerde tekrar edebilir. crop_count tekil plaka sayısı değildir.
    crops: kaynak saniyesi, frame_index, piksel bbox_xyxy (x2/y2 hariç),
    confidence ve yerel crop_path içerir; ilk 30 gösterilir, tümü details_path
    dosyasındadır. Tespit yokluğu plaka yokluğunu kanıtlamaz. Dosya dışarı
    gönderilmez, kaynak değiştirilmez. Ortak tool JSON zarfını döndürür.
    """
    from utils.plate_detection import PlateError, extract_plate_crops

    try:
        if not video_path or not os.path.isfile(video_path):
            return _tool_error("FILE_NOT_FOUND", "Kaynak video bulunamadı.")
        start, end, metadata, clamped = _validate_video_range(video_path, start_sec, end_sec)
        data, warnings = extract_plate_crops(video_path, start, end)
        if clamped:
            warnings.append({"code": "END_TIME_CLAMPED", "message": "Bitiş video sonunda sınırlandırıldı."})
        return _tool_result(ok=True, data={**data, "video": metadata,
            "requested_range": {"start_sec": start_sec, "end_sec": end_sec}}, warnings=warnings)
    except (ToolInputError, PlateError) as exc:
        return _tool_error(exc.code, str(exc), data=exc.data)
    except ImportError as exc:
        return _tool_error("PLATE_DEPENDENCY_MISSING", f"requirements-plates.txt bağımlılığı eksik: {exc}")
    except Exception as exc:
        return _tool_error("PLATE_DETECTION_ERROR", f"{type(exc).__name__}: {exc}")


@tool
def archive_anomaly_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    category: ArchiveCategory,
    explanation: str,
) -> str:
    """İlgili olay kesitini seçilen kategori altında yerel arşive kaydeder.

    Kategori görsel kanıta göre çağıran agent tarafından seçilir: hirsizlik,
    soygun, kavga_saldiri, trafik_kazasi, is_kazasi, diger, belirsiz.
    Olay türü biliniyor ama listede yoksa diger; türü anlaşılmıyorsa belirsiz.
    Kendisi olay sınıflandırmaz; explanation kanıta dayalı kısa gerekçedir.
    Tek kesit tek kategoriye kaydedilir; source video saniyeleri kullanılır.
    Bitiş video sonuna kırpılabilir, geçersiz başlangıç reddedilir.
    _stuff/lab_runs/actions/archive/<category>/ altında MP4 ve metadata JSON
    oluşturur. FFmpeg yeniden kodlama yapar; genel save_video_segment aracından
    farklı olarak keyframe öncesini taşımayan kesim hedeflenir, kare hassasiyetindedir.
    Aynı kaynak/aralık/kategori yeniden istenirse geçerli mevcut klip kullanılır;
    cache_hit=True, ilk gerekçe korunur. Bozuk/eski kaydın üzerine yazılmaz.
    Çıktı yolları output_path/metadata_path'tir. Kaynak değiştirilmez, dışarıya
    aktarılmaz. Bu bir arşivleme eylemidir; raporun kendisini üretmez.
    """
    try:
        if not video_path or not os.path.isfile(video_path):
            return _tool_error("FILE_NOT_FOUND", "Kaynak video bulunamadı.")
        start, end, metadata, clamped = _validate_video_range(video_path, start_sec, end_sec)
        data = archive_clip(video_path, start, end, category, explanation)
        warnings = []
        if clamped:
            warnings.append({"code": "END_TIME_CLAMPED", "message": "Bitiş video sonunda sınırlandırıldı."})
        return _tool_result(ok=True, data={**data, "video": metadata,
            "requested_range": {"start_sec": start_sec, "end_sec": end_sec}}, warnings=warnings)
    except (ToolInputError, ArchiveError) as exc:
        return _tool_error(exc.code, str(exc), data=exc.data)
    except VideoExportError as exc:
        return _tool_error(exc.code, str(exc))
    except FileNotFoundError:
        return _tool_error("FFMPEG_NOT_FOUND", "FFmpeg veya kaynak dosya bulunamadı.")
    except subprocess.TimeoutExpired:
        return _tool_error("FFMPEG_TIMEOUT", "FFmpeg 300 saniyede tamamlanamadı.")
    except subprocess.CalledProcessError as exc:
        return _tool_error("FFMPEG_ERROR", (exc.stderr or "FFmpeg başarısız.")[-500:])
    except Exception as exc:
        return _tool_error("ARCHIVE_ERROR", f"{type(exc).__name__}: {exc}")


@tool
def read_license_plate_crops(crops_manifest_path: str) -> str:
    """Plaka tespitinin mevcut kırpımlarındaki yazıyı yerel OCR modeliyle okur.

    crops_manifest_path: detect_license_plate_regions sonucundaki details_path
    (crops.json). Videoyu yeniden taramaz, plaka tespitini tekrarlamaz. Özet
    listedeki ilk 30 ile sınırlı değildir; kayıt dosyasındaki tüm kırpımları okur.
    text yalnız status=read olduğunda doludur; uncertain/unreadable sonuçlarda
    text=null, ham tahmin candidate_text içindedir ve kesin plaka diye sunulmaz.
    min_slot_confidence karakter/sonlandırma yuvalarının en düşük model güvenidir;
    doğruluk olasılığı değildir. Yüksek güvenli okumalar da gözle doğrulanmalıdır.
    Kaynak saniyesi, koordinat, crop_path ve tespit güveni her okumada korunur.
    Aynı plakanın farklı karelerdeki okumaları birleştirilmez; sayılar tekil araç
    sayısı değildir. Sahip/kimlik sorgusu, ülke tahmini veya dışarı aktarım yapmaz.
    İlk 30 sonuç özettedir, tümü yeni details_path JSON dosyasındadır. Eksik/bozuk
    kırpım veya limit aşımında hata döner; kısmi sonuç başarı diye sunulmaz.
    Sonuç ortak {ok, data, warnings, error} JSON zarfındadır.
    """
    from utils.plate_detection import PlateError
    from utils.plate_ocr import read_plate_crops

    try:
        data, warnings = read_plate_crops(crops_manifest_path)
        return _tool_result(ok=True, data=data, warnings=warnings)
    except PlateError as exc:
        return _tool_error(exc.code, str(exc), data=exc.data)
    except ImportError as exc:
        return _tool_error("OCR_DEPENDENCY_MISSING", f"requirements-plates.txt bağımlılığı eksik: {exc}")
    except Exception as exc:
        return _tool_error("PLATE_OCR_ERROR", f"{type(exc).__name__}: {exc}")


tools = [
    run_abnormal_event_segmenter, 
    analyze_video_with_vlm, 
    save_video_segment, 
    get_video_info,
    detect_and_track_objects,
    detect_license_plate_regions,
    read_license_plate_crops,
    archive_anomaly_clip,
]
