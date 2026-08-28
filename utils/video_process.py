from __future__ import annotations

import gc
import os
import cv2
import torch
import numpy as np
from typing import List, Optional

from decord import VideoReader, cpu


def get_report_dir(video_path: str, base_dir: str = "ABNORMAL_DETECTION") -> str:
    """
    Video adina gore, raporun tum parcalarinin (grafik, klipler, report.json)
    kaydedilecegi ortak klasoru dondurur, boylece hepsi ayni yerde toplanir.

    Ornek: 'videos/test_video_4.mp4' -> 'ABNORMAL_DETECTION/test_video_4'

    Not: `SegmentRankingModel.to_segment` da grafigi kaydederken bu fonksiyonu
    kullanir; boylece test.py'de kliplerin/raporun ayni klasore yazildigindan
    emin olmak icin sadece bu fonksiyonu cagirmak yeterlidir.
    """
    os.makedirs(base_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(base_dir, file_name)


def save_segment_clips(video_path: str,
                       segments: list[dict],
                       save_dir: str,
                       prefix: str = "segment") -> List[Optional[str]]:
    """
    Verilen `segments` listesindeki (start_time, end_time) araliklarina gore
    orijinal videodan kirpilmis klipleri `save_dir` klasorune .mp4 olarak kaydeder.

    Not: OpenCV'nin 'mp4v' codec'i ekstra bir bagimlilik (ffmpeg binary'si)
    gerektirmez, fakat H.264 kadar genis tarayici/oynatici uyumlulugu
    saglamayabilir. Daha genis uyumluluk gerekiyorsa ffmpeg ile yeniden
    kodlamak (transcode) dusunulebilir.

    Args:
        video_path: Orijinal video dosyasinin yolu.
        segments: `SegmentRankingModel.to_segment` ciktisi (start_time, end_time, duration).
        save_dir: Kliplerin kaydedilecegi klasor (yoksa olusturulur).
        prefix: Klip dosya adlarinin on eki (ornek: 'segment_01.mp4').

    Returns:
        list[str | None]: `segments` ile ayni sirada, save_dir icindeki klip
        dosya adlari (basename, tam yol degil). Bir segment kirpilamazsa
        (ornegin video okunamazsa) None doner.
    """
    if not segments:
        return []

    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc       = cv2.VideoWriter_fourcc(*"mp4v")

    clip_paths = []

    for idx, segment in enumerate(segments, start=1):
        start_frame = max(0, int(segment["start_time"] * fps))
        end_frame   = min(max(0, total_frames - 1), int(segment["end_time"] * fps))

        clip_name = f"{prefix}_{idx:02d}.mp4"
        clip_path = os.path.join(os.getcwd(), save_dir, clip_name)

        writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        wrote_any_frame = False
        for _ in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            wrote_any_frame = True

        writer.release()

        if wrote_any_frame:
            clip_paths.append(clip_path)
        else:
            if os.path.exists(clip_path):
                os.remove(clip_path)
            clip_paths.append(None)

    cap.release()
    return clip_paths


def get_video_length(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    seconds_of_video = original_number_of_frames/original_fps

    return seconds_of_video


def fetch_video_patches(video_path: str, 
                        target_fps: int = 30, 
                        patch_size: int = 32, 
                        resize_dim: tuple = (112, 112),
                        crop_dim: tuple = (112, 112),
                        clip_size: int = 16,
                        max_sec: float = 241.5):
    
    vr = VideoReader(video_path, ctx=cpu(0), width=resize_dim[1], height=resize_dim[0])
    
    original_fps = vr.get_avg_fps()
    original_number_of_frames = len(vr)
    
    ratio = target_fps / original_fps if original_fps > 0 else 1.0
    target_number_of_frames = int(original_number_of_frames * ratio)

    duration = original_number_of_frames / original_fps if original_fps > 0 else 0
    patch_size_K = max(1.0, duration / max_sec)
    dynamic_patch_size = int(patch_size * patch_size_K)

    number_of_patch_elements = max(1, target_number_of_frames // dynamic_patch_size)
    clean_target_frames = number_of_patch_elements * dynamic_patch_size
    
    target_indices = np.arange(clean_target_frames)
    needed_orig_indices = (target_indices / ratio).astype(int)
    needed_orig_indices = np.clip(needed_orig_indices, 0, original_number_of_frames - 1)

    for start_idx in range(0, clean_target_frames, number_of_patch_elements):
        
        batch_indices = needed_orig_indices[start_idx : start_idx + number_of_patch_elements]
        frames = vr.get_batch(batch_indices).asnumpy() 
        
        if crop_dim:
            h, w = frames.shape[1:3]
            crop_h, crop_w = crop_dim
            start_y = (h - crop_h) // 2
            start_x = (w - crop_w) // 2
            frames = frames[:, start_y:start_y+crop_h, start_x:start_x+crop_w, :]
            
        batch_tensor = process_patch_fast(frames, clip_size)
        yield batch_tensor

    del vr, frames
    gc.collect()

def process_patch_fast(frames_array, clip_size=16):
    
    tensor = torch.from_numpy(frames_array)  # uint8
    tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 4, 1, 2, 3).contiguous()  # (1, 3, T, H, W) uint8
    
    T = tensor.shape[2]
    if T < clip_size:
        pad_size = clip_size - T
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, pad_size))
        
    return tensor

def generate_frames(video_path: os.PathLike, 
                    start_sec: float, 
                    end_sec: float, 
                    all_video: bool = False, 
                    FPS: float = 5,
                    max_frames: int = 32):

    # Probe one native frame before decoding the batch at a bounded resolution.
    # A fixed landscape size distorts portrait and widescreen source videos.
    vr = VideoReader(video_path, ctx=cpu(0))
    source_height, source_width = vr.get_batch([0]).asnumpy()[0].shape[:2]
    scale = min(1.0, 448 / max(source_width, source_height))
    width = max(1, round(source_width * scale))
    height = max(1, round(source_height * scale))
    del vr
    vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    total_sec = total_frames / video_fps
    
    if all_video:
        start_sec = 0.0
        end_sec = total_sec
    
    start_frame_idx = max(0, int((start_sec / total_sec) * total_frames))
    end_frame_idx   = min(total_frames - 1, int((end_sec / total_sec) * total_frames))
    
    duration = end_sec - start_sec
    calculated_frames = int(duration * FPS)
    
    frame_length = max(1, min(max_frames, calculated_frames))
    frames_ids = np.linspace(start_frame_idx, 
                             end_frame_idx, 
                             frame_length, 
                             dtype=int)
    
    actual_start_sec = frames_ids[0] / video_fps
    actual_end_sec = frames_ids[-1] / video_fps
    frames = vr.get_batch(frames_ids).asnumpy()
    
    return frames, actual_start_sec, actual_end_sec
