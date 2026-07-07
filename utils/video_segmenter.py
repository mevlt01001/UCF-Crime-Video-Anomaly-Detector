import os
import random
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .video_preprocess import fetch_video_patches

class MILRankingNetwork(nn.Module):
    def __init__(self, input_dim=4096): # Sultani modeliyle uyumlu boyut
        super(MILRankingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)

class VideoSegmenterLoss(nn.Module):
    def __init__(self, lambda_1=0, lambda_2=0):
        super(VideoSegmenterLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(self, y_anomaly, y_normal):
        """
        y_anomaly.shape : [B, 32]
        y_normal.shape : [B, 32]
        """
        max_anomaly, _ = torch.max(y_anomaly, dim=1)  
        max_normal, _ = torch.max(y_normal, dim=1)    
        
        hinge_loss = F.relu(1.0 - max_anomaly + max_normal)
        
        smoothness = torch.sum((y_anomaly[:, :-1] - y_anomaly[:, 1:]) ** 2, dim=1)
        sparsity = torch.sum(y_anomaly, dim=1)
        
        mean_hinge = torch.mean(hinge_loss)
        mean_smoothness = self.lambda_1 * torch.mean(smoothness)
        mean_sparsity = self.lambda_2 * torch.mean(sparsity)
        
        return mean_hinge + mean_smoothness + mean_sparsity

def _probe_segment_length(video_path: str, target_fps: int, patch_size: int):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if original_fps <= 0 or original_frames <= 0:
        return None

    ratio = target_fps / original_fps
    target_frames = int(original_frames * ratio)
    S = target_frames // patch_size  # bir segmentteki frame sayisi
    return S


def _num_clips_for_segment(S: int, clip_size: int, stride: int) -> int:
    S_eff = max(S, clip_size)  # model kisa segmentleri clip_size'a pad'liyor
    return (S_eff - clip_size) // stride + 1


def _estimate_conv1_bytes(num_clips_in_batch: int, resize_dim: tuple,
                           clip_size: int, dtype_bytes: int = 4,
                           conv1_channels: int = 64) -> int:
    H, W = resize_dim
    bytes_per_clip = conv1_channels * clip_size * H * W * dtype_bytes
    return bytes_per_clip * num_clips_in_batch


@torch.no_grad()
def extract_C3D_features(extractor,
                          video_paths: list,
                          save_dir: str,
                          patch_size: int = 32,
                          batch: int = 4,
                          resize_dim: tuple = (112, 112),
                          target_fps: int = 30,
                          safe_vram_gb: float = 3.0,
                          skip_log_path: str = None):

    os.makedirs(save_dir, exist_ok=True)
    if skip_log_path is None:
        skip_log_path = os.path.join(save_dir, "skipped_videos_vram.txt")

    is_dp = isinstance(extractor, torch.nn.DataParallel)
    core_model = extractor.module if is_dp else extractor
    clip_size = core_model.clip_size
    stride = core_model.stride

    if not is_dp:
        extractor = extractor.to("cuda")
    extractor.eval()

    safe_budget_bytes = safe_vram_gb * (1024 ** 3)
    bytes_per_clip = _estimate_conv1_bytes(1, resize_dim, clip_size)

    skipped = []

    for vp in tqdm(video_paths):
        save_path = os.path.join(save_dir, os.path.basename(vp) + ".pt")
        if os.path.exists(save_path):
            continue

        S = _probe_segment_length(vp, target_fps, patch_size)
        if S is None:
            print(f"[SKIP] {vp}: video metadata okunamadi")
            skipped.append((vp, "metadata okunamadi"))
            continue

        num_clips = _num_clips_for_segment(S, clip_size, stride)
        single_segment_bytes = bytes_per_clip * num_clips

        if single_segment_bytes > safe_budget_bytes:
            est_gb = single_segment_bytes / (1024 ** 3)
            msg = (f"~{num_clips} clips per segment, ~{est_gb:.2f} GB Estimated"
                   f"(Budget: {safe_vram_gb:.2f} GB) -- VRAM'i booming, skipped.")
            print(f"[SKIP] {vp}: {msg}")
            skipped.append((vp, msg))
            continue

        effective_batch = max(1, min(batch, int(safe_budget_bytes // single_segment_bytes)))

        video_feature_list = []
        mini_batch_buffer = []
        num_segments_seen = 0

        def flush_buffer():
            if not mini_batch_buffer:
                return
            mini_batch = torch.cat(mini_batch_buffer, dim=0)  # (b, 3, S, H, W) uint8
            mini_batch_buffer.clear()

            mini_batch = mini_batch.to("cuda", non_blocking=True)
            mini_batch = mini_batch.float().div_(255.0)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                mini_feat = extractor(mini_batch)  # (b, 8192)

            video_feature_list.append(mini_feat.detach().cpu().float())
            del mini_batch, mini_feat

        for segment in fetch_video_patches(vp, target_fps=target_fps, patch_size=patch_size, resize_dim=resize_dim):
            mini_batch_buffer.append(segment)
            num_segments_seen += 1

            if len(mini_batch_buffer) == effective_batch:
                flush_buffer()

        flush_buffer()

        if num_segments_seen != patch_size:
            print(f"[SKIP] {vp}: Expected {patch_size} segment, got {num_segments_seen}")
            torch.cuda.empty_cache()
            continue

        video_feature = torch.cat(video_feature_list, dim=0)  # (patch_size, 8192)
        torch.save(video_feature, save_path)

        del video_feature_list, video_feature
        torch.cuda.empty_cache()

    if skipped:
        with open(skip_log_path, "w") as f:
            for vp, reason in skipped:
                f.write(f"{vp}\t{reason}\n")
        print(f"\nNumber of {len(skipped)} videos skipped due to VRAM -> {skip_log_path}")


def MIL_network_trainer(model: MILRankingNetwork, 
                        anormal_feat_dir: str, 
                        normal_feat_dir: str, 
                        epochs: int=10, 
                        learning_rate: float=0.001,
                        batch_size: int=16): # Batch parametresi eklendi
    
    model = model.train(True)
    
    criterion = VideoSegmenterLoss(lambda_1=8e-5, lambda_2=8e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, weight_decay=0.01)

    anormal_files = [os.path.join(anormal_feat_dir, f) for f in os.listdir(anormal_feat_dir) if f.endswith('.pt')]
    normal_files = [os.path.join(normal_feat_dir, f) for f in os.listdir(normal_feat_dir) if f.endswith('.pt')]

    best_loss = float("inf")

    for epoch_idx in range(epochs):
        random.shuffle(anormal_files)
        
        epoch_loss = 0.0
        
        # Toplam batch sayısını hesaplıyoruz
        num_batches = (len(anormal_files) + batch_size - 1) // batch_size

        for i in range(num_batches):
            optimizer.zero_grad()
            # O anki batch için anormal dosyaları alıyoruz
            batch_anormal_files = anormal_files[i * batch_size : (i + 1) * batch_size]
            current_b_size = len(batch_anormal_files)
            
            # Aynı sayıda rastgele normal dosya seçiyoruz
            batch_normal_files = random.choices(normal_files, k=current_b_size)

            # Dosyaları yükleyip torch.stack ile birleştiriyoruz (B, 32, 8192)
            feat_anomaly = torch.stack([torch.load(f) for f in batch_anormal_files]).to("cuda")
            feat_normal = torch.stack([torch.load(f) for f in batch_normal_files]).to("cuda")
            
            y_anomaly = model.forward(feat_anomaly) # [B, 32]
            y_normal = model.forward(feat_normal)   # [B, 32]

            # print(f"normal_video_segment scores: {y_normal}")
            # print(f"anormal_video_segment scores: {y_anomaly}")
        
            loss = criterion.forward(y_anomaly, y_normal)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            progress_percent = ((i + 1) / num_batches) * 100
            print(f"Epoch {epoch_idx+1:03d}/{epochs} - Progress: %{progress_percent:5.3f} - Loss: {loss.item():.6f}", end="\r")
        
        print()
        
        avg_loss = epoch_loss / num_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best.pt")

        torch.save(model.state_dict(), "last.pt")