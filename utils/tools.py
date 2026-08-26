from __future__ import annotations

import os
import gc
import json
import subprocess
import cv2
import torch
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from langchain.tools import tool
from decord import VideoReader, cpu
from typing import Generator, Optional

from utils.env import env_first, env_float, env_get, env_int
from utils.video_analyzer_model import Video_Analyzer, pick_device
from utils.video_process import generate_frames
from utils.vlm import VLM_MAX_FRAMES, VLM_Manager

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



FILE_NOT_FOUND_ERROR = lambda f: f"Beliritlen {f} dosyası bulunamadı"
VIDEO_TOO_SHORT_ERROR = lambda vp, ms, vs : f"Belirtilen {vp} dosyası video analiz segmentasyonu gerçekleştirebilemk için çok kısa! Lütfen {ms} saniyeden daha fazla olan video ile deneyin. Sizin videonuz {vs} saniye yalnızca."

video_cache: dict[str, VideoReader] = {}

def get_vr(video_path: str) -> VideoReader:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")
    if video_path not in video_cache:
        video_cache[video_path] = VideoReader(video_path, ctx=cpu(0))
    return video_cache[video_path]

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
            for end_idx in range(clip_size, total_frames, stride):
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
    """
    Bu araç, belirtilen videoyu yapay zeka anomali tespit modelinden geçirir.
    Kavga, kaza, hırsızlık, yangın veya anormal insan/araç hareketlerini tespit etmek için KESİNLİKLE bu aracı kullanın.
    Çıktı olarak anormal anların zaman damgalarını (timestamp) ve ilgili video kesitlerinin yollarını döner.
    """
    if not video_path or not os.path.exists(video_path):
        return FILE_NOT_FOUND_ERROR(video_path)

    try:
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
        if not segments:
            return (
                f"Analiz tamamlandı. {video_path} içinde eşik üstü anormal segment bulunamadı "
                "(threshold=0.3). Video normal görünebilir veya skorlar eşiğin altında kalmış olabilir."
            )
        return json.dumps(
            {
                "video_path": video_path,
                "segment_count": len(segments),
                "segments": segments,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        print(f"[segmenter hata] {type(e).__name__}: {e}")
        return f"Got an error [{video_path}]: {type(e).__name__}: {e}"

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
    """
    Bu araç, Vision-Language Model (VLM) kullanarak videonun belirli bir zaman aralığındaki olayları anlamlandırır, 
    görsel soru-cevap yapar, metin okur (OCR) veya iki sahneyi karşılaştırır.
    
    Parametreler:
    - video_path: İncelenecek videonun yolu.
    - query: VLM'e sorulacak soru (Örn: "Adam ne renk ceket giyiyor?", "Sandalyeden kalkan oldu mu?").
    - start_sec: İncelemenin başlayacağı saniye (float). Başlangıç için 0.0 kullan.
    - end_sec: İncelemenin biteceği saniye (float).
    """
    if start_sec >= end_sec:
        return "Hata: start_sec, end_sec'den küçük olmalıdır."

    try:
        vlm = VLM_Manager(
            api_key=EVREN_API_KEY,
            base_url=EVREN_URL,
            system_prompt=VLM_SYSTEM_PROMT,
        )
        frames, actual_start, actual_end = generate_frames(
            video_path=video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            all_video=False,
            FPS=VLM_SOURCE_SAMPLE_FPS,
            max_frames=VLM_MAX_FRAMES
        )
        source_time_context = (
            "ZAMAN REFERANSI (zorunlu): İncelediğin geçici klip, kaynak videonun "
            f"{actual_start:.2f}–{actual_end:.2f} saniye aralığından örneklenmiştir. "
            f"Geçici klipte 0:00, kaynak videoda {actual_start:.2f}. saniyeye karşılık gelir. "
            "Yanıtında zaman belirteceksen yalnızca kaynak video zamanını kullan; "
            "geçici klibin 0:00, başlangıç veya ilk saniye ifadelerini kaynak videonun "
            "başlangıcı gibi sunma.\n\n"
            f"Kullanıcı sorusu: {query}"
        )
        # Kare zamanları FPS yuvarlaması nedeniyle birkaç salise kayabilir; geçici
        # MP4 süresini kullanıcının/segmenterin istediği gerçek aralığa sabitle.
        source_duration = max(0.0, end_sec - start_sec)
        response = vlm.run(
            text=source_time_context,
            frames=frames,
            source_duration=source_duration,
        )
        
        del frames
        gc.collect()
        
        return (
            f"VLM Analiz Sonucu (kaynak video {actual_start:.2f}sn - "
            f"{actual_end:.2f}sn aralığı için; geçici klip 0:00 = kaynak "
            f"{actual_start:.2f}sn): {response}"
        )
        
    except Exception as e:
        return f"VLM analizi sırasında hata oluştu: {str(e)}"

@tool
def save_video_segment(video_path: str, start_sec: float, end_sec: float, output_filename: str) -> str:
    """
    Bu araç, videonun belirli bir zaman aralığını (start_sec ile end_sec arası) kesip yeni bir video dosyası olarak kaydeder.
    Kullanıcı "şu saniyeleri benim için kaydet" veya "şu anı kes" dediğinde KESİNLİKLE bu aracı kullanın.
    
    Parametreler:
    - video_path: Orijinal videonun yolu.
    - start_sec: Kesimin başlayacağı saniye (Örn: 2dk 13sn -> 133.0).
    - end_sec: Kesimin biteceği saniye.
    - output_filename: Kaydedilecek dosyanın adı (örn: "kesilmiş_olay.mp4").
    """
    if not video_path or not os.path.isfile(video_path):
        return f"Video kesme işlemi başarısız: kaynak video bulunamadı: {video_path}"
    if start_sec < 0:
        return "Video kesme işlemi başarısız: start_sec negatif olamaz."
    if end_sec <= start_sec:
        return "Video kesme işlemi başarısız: end_sec, start_sec değerinden büyük olmalıdır."
    if not (output_filename or "").strip():
        return "Video kesme işlemi başarısız: output_filename boş olamaz."

    output_path = Path(output_filename.strip()).expanduser()
    if output_path.suffix.lower() != ".mp4":
        output_path = Path(f"{output_path}.mp4")
    if not output_path.parent.is_dir():
        return f"Video kesme işlemi başarısız: çıktı klasörü bulunamadı: {output_path.parent}"

    try:
        command = [
            'ffmpeg', '-y', 
            '-ss', str(start_sec), 
            '-i', video_path, 
            '-t', str(end_sec - start_sec), 
            '-c:v', 'copy', '-c:a', 'copy',
            str(output_path),
        ]

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if not output_path.is_file():
            return "Video kesme işlemi başarısız: FFmpeg tamamlandı ancak çıktı dosyası oluşmadı."
        if output_path.stat().st_size <= 0:
            return "Video kesme işlemi başarısız: çıktı dosyası boş."

        cap = cv2.VideoCapture(str(output_path))
        try:
            opened = cap.isOpened()
            has_frame, _ = cap.read() if opened else (False, None)
        finally:
            cap.release()
        if not opened or not has_frame:
            return "Video kesme işlemi başarısız: çıktı dosyası geçerli bir video olarak okunamadı."

        return (
            f"Başarılı! Video {start_sec} - {end_sec} saniyeleri arası kesilip "
            f"'{output_path}' adıyla kaydedildi."
        )
    except FileNotFoundError:
        return "Video kesme işlemi başarısız: FFmpeg bulunamadı veya çalıştırılamadı."
    except subprocess.TimeoutExpired:
        return "Video kesme işlemi başarısız: FFmpeg 300 saniye içinde tamamlanamadı."
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or "FFmpeg hata ayrıntısı döndürmedi.").strip()
        if len(detail) > 500:
            detail = detail[-500:]
        return f"Video kesme işlemi başarısız: FFmpeg hata verdi: {detail}"
    except Exception as e:
        return f"Video kesme işlemi başarısız: {str(e)}"

@tool
def get_video_info(video_path: str) -> str:
    """
    Videonun toplam uzunluğunu (saniye), FPS değerini ve çözünürlüğünü döndürür.
    Zaman hesaplamaları yapmadan önce videonun toplam süresini öğrenmek için kullanın.
    """
    try:
        vr = get_vr(video_path)
        fps = vr.get_avg_fps()
        total_seconds = len(vr) / fps
        shape = vr[0].shape # (H, W, C)
        return f"Video Bilgisi: Toplam {total_seconds:.2f} saniye, {fps:.2f} FPS, Çözünürlük: {shape[1]}x{shape[0]} (WxH)."
    except Exception as e:
        return f"Video bilgisi alınamadı: {str(e)}"

tools = [
    run_abnormal_event_segmenter, 
    analyze_video_with_vlm, 
    save_video_segment, 
    get_video_info
]
