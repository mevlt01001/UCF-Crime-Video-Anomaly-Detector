import os
import cv2
import torch
import numpy as np
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
                       prefix: str = "segment") -> list[str | None]:
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

    clip_names = []

    for idx, segment in enumerate(segments, start=1):
        start_frame = max(0, int(segment["start_time"] * fps))
        end_frame   = min(max(0, total_frames - 1), int(segment["end_time"] * fps))

        clip_name = f"{prefix}_{idx:02d}.mp4"
        clip_path = os.path.join(save_dir, clip_name)

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
            clip_names.append(clip_name)
        else:
            if os.path.exists(clip_path):
                os.remove(clip_path)
            clip_names.append(None)

    cap.release()
    return clip_names


def get_video_lenght(video_path: str):
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
                        clip_size: int = 16):
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    ratio = target_fps / original_fps

    original_number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_number_of_frames = int(original_number_of_frames * ratio)

    number_of_patch_elements = max(1, target_number_of_frames // patch_size)
    clean_target_frames = number_of_patch_elements * patch_size
    
    frames = []
    current_orig_idx = -1
    last_frame = None

    for target_idx in range(clean_target_frames):
        needed_orig_idx = int(target_idx / ratio)

        while current_orig_idx < needed_orig_idx - 1:
            if not cap.grab():
                break
            current_orig_idx += 1

        if current_orig_idx < needed_orig_idx:
            ret, frame = cap.read()
            if ret:
                current_orig_idx += 1
                last_frame = frame

        if last_frame is None:
            break
            
        processed_frame = last_frame.copy()
        if resize_dim: 
            processed_frame = cv2.resize(processed_frame, resize_dim)
            
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        frames.append(processed_frame)    
        
        if len(frames) == number_of_patch_elements:
            batch_tensor = process_patch(frames, clip_size)
            yield batch_tensor
            frames = []

    cap.release()

def process_patch(frames_list, clip_size=16):
    np_frames = np.array(frames_list)  # (T, H, W, 3) uint8
    tensor = torch.from_numpy(np_frames)  # uint8
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
                    frame_length: int = 16):

    vr = VideoReader(video_path, ctx=cpu(0), width=320, height=240)

    total_frames = len(vr)
    total_sec    = total_frames / vr.get_avg_fps()
    
    start_frame_idx = max(0, int((start_sec / total_sec) * total_frames))
    end_frame_idx   = min(total_frames - 1, int((end_sec / total_sec) * total_frames)) if not all_video else total_frames - 1
    
    frames_ids = np.linspace(start_frame_idx, 
                                end_frame_idx, 
                                frame_length, 
                                dtype=int)
    
    start_sec = total_sec*(start_frame_idx/total_frames)
    end_sec = total_sec*(end_frame_idx/total_frames)
    
    frames = vr.get_batch(frames_ids).asnumpy()
    
    return frames, start_sec, end_sec