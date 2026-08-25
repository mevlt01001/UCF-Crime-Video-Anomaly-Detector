import os
import gc
import torch
import numpy as np

from tqdm import tqdm
from dotenv import load_dotenv
from langchain.tools import tool
from decord import VideoReader, cpu
from typing import Generator, Tuple

from utils import Video_Analyzer, get_video_length, VLM_Manager, generate_frames
load_dotenv()

AS_MODEL_NAME = os.environ.get("AS_MODEL_NAME")
AS_OVERLAP = os.environ.get("AS_OVERLAP")
AS_FC_CHECKPOINT = os.environ.get("AS_FC_CHECKPOINT")
AS_TRT_PLAN_PATH = os.environ.get("AS_TRT_PLAN_PATH")
AS_BATCH = os.environ.get("AS_BATCH")
EVREN_API_KEY = os.environ.get("EVREN_API_KEY")
EVREN_URL = os.environ.get("EVREN_URL")
VLM_SYSTEM_PROMT = os.environ.get("VLM_SYSTEM_PROMT")
AS_CLIP_SIZE = os.environ.get("AS_CLIP_SIZE")
AS_STRIDE = os.environ.get("AS_STRIDE")
AS_FPS = os.environ.get("AS_FPS")
AS_WIDTH = os.environ.get("AS_WIDTH")
AS_HEIGHT = os.environ.get("AS_HEIGHT")


# ANOMALY SEGMETNER MODEL
anomaly_segment_model = Video_Analyzer(
    AS_MODEL_NAME, AS_CLIP_SIZE, 
    AS_OVERLAP, AS_FC_CHECKPOINT
).eval()

anomaly_segment_model.export_trt(AS_TRT_PLAN_PATH, 
                                 AS_BATCH, (AS_HEIGHT, AS_WIDTH))
# ANOMALY SEGMETNER MODEL

# VISION LANGUAGE MODEL
vlm = VLM_Manager(
    api_key=EVREN_API_KEY,
    base_url=EVREN_URL,
    system_prompt=VLM_SYSTEM_PROMT
)
# VISION LANGUAGE MODEL

FILE_NOT_FOUND_ERROR = lambda f: f"Beliritlen {f} dosyası bulunamadı"
VIDEO_TOO_SHORT_ERROR = lambda vp, ms, vs : f"Belirtilen {vp} dosyası video analiz segmentasyonu gerçekleştirebilemk için çok kısa! Lütfen {ms} saniyeden daha fazla olan video ile deneyin. Sizin videonuz {vs} saniye yalnızca."

video_cache: dict[str, VideoReader] = {}

def get_vr(video_path: str) -> VideoReader:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")
    if video_path not in video_cache:
        video_cache[video_path] = VideoReader(video_path, ctx=cpu(0))
    return video_cache[video_path]

def create_clip_generator(
        video_path: os.PathLike,
        clip_size: int,
        stride: int,
        fps: int,
        width: int,
        height: int) -> tuple[Generator[np.array, None, None], int]|str:

    vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height, num_threads=2)
    org_fps = vr.get_avg_fps()
    frame_indices = np.arange(start=0, stop=len(vr), step=vr.get_avg_fps()/fps).long()
    total_frames = len(frame_indices)

    video_seconds = org_fps*len(vr)
    minimum_seconds = clip_size/fps

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

    out = create_clip_generator(
        video_path,
        AS_CLIP_SIZE,
        AS_STRIDE,
        AS_FPS,
        AS_WIDTH,
        AS_HEIGHT
    )
    
    if isinstance(str, out):
        return out

    clip_generator, total_clips = out

    mini_batch = [] 
    scores = [] 
    pbar = tqdm(
        total=total_clips, 
        desc=f"Processing {os.path.basename(video_path)} for abnormal events"
    )

    def inference(mini_batch):
        batch_tensor = torch.stack(mini_batch).to("cuda")
        batch_tensor = anomaly_segment_model.preprocess(batch_tensor)
        batch_scores = anomaly_segment_model.forward(batch_tensor).detach().cpu().float()
        scores.append(batch_scores)
    
    try:
        for clip in clip_generator: # [T, H, W, C]
            mini_batch.append(clip)
            
            if len(mini_batch) == AS_BATCH:
                inference(mini_batch)
                pbar.update(len(mini_batch))

                mini_batch = []
                torch.cuda.empty_cache()

        if len(mini_batch) > 0:
            inference(mini_batch)
            pbar.update(len(mini_batch))

    except Exception as e:
        pbar.close()
        return f"\nGot an error [{video_path}]: {e}"
    finally:
        pbar.close()

    return anomaly_segment_model.create_clips(
        torch.concat(scores, dim=0), 
        get_video_length(video_path),
        video_path,"save_dir",
        0.5,3.0,2.0,False,False
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
        frames, actual_start, actual_end = generate_frames(
            video_path=video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            all_video=False,
            FPS=5,
            max_frames=128
        )
        
        response = vlm.run(query=query, frames=frames)
        
        del frames
        gc.collect()
        
        return f"VLM Analiz Sonucu ({actual_start:.2f}sn - {actual_end:.2f}sn aralığı için): {response}"
        
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
    try:
        if not output_filename.endswith('.mp4'):
            output_filename += '.mp4'
            
        import subprocess
        command = [
            'ffmpeg', '-y', 
            '-ss', str(start_sec), 
            '-i', video_path, 
            '-t', str(end_sec - start_sec), 
            '-c:v', 'copy', '-c:a', 'copy',
            output_filename
        ]
        
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Başarılı! Video {start_sec} - {end_sec} saniyeleri arası kesilip '{output_filename}' adıyla kaydedildi."
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