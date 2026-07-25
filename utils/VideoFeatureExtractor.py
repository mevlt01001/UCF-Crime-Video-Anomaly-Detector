import os
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .video_preprocess import fetch_video_patches
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights

class FeatureExtractor(nn.Module):
    def __init__(self, clip_size: int = 16, overlap: int = 0, max_frames: int = 54000):
        super(FeatureExtractor, self).__init__()
        self.clip_size = clip_size
        self.overlap = overlap
        self.stride = clip_size - overlap

        weights = S3D_Weights.DEFAULT
        self.backbone = s3d(weights=weights)
        self.backbone.classifier = nn.Identity()

        max_clips = (max_frames - clip_size) // self.stride + 1
        self.indices = torch.zeros(max_clips, clip_size, dtype=torch.long)
        for i in range(max_clips):
            self.indices[i] = torch.arange(i * self.stride, i * self.stride + clip_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, S, H, W = x.shape

        num_clips = (S - self.clip_size) // self.stride + 1
        
        valid_indices = self.indices[:num_clips] 
        flat_indices = valid_indices.flatten()

        x_clips = x.index_select(dim=2, index=flat_indices.to(x.device)) 
        
        x_clips = x_clips.view(B, C, num_clips, self.clip_size, H, W) # [B, C, S, ClipSize, H, W]
        x_clips = x_clips.permute(0, 2, 1, 3, 4, 5) # [B, S, C, ClipSize, H, W]

        h = x_clips.reshape(-1, C, self.clip_size, H, W) # [B*S, C, ClipSize, H, W]
        h = self.backbone(h) # [B*S, Dim]
        h = h.reshape(B, num_clips, -1)
        h = h.mean(dim=1)  

        return h

def calc_vram_usage_in_mb(model: nn.Module, tensor: torch.Tensor) -> float:

    device = "cuda"
    tensor = tensor.to(device, non_blocking=True)
    
    torch.cuda.reset_peak_memory_stats(device)
    initial_mem = torch.cuda.memory_allocated(device)
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        _ = model(tensor)
        
    peak_mem = torch.cuda.max_memory_allocated(device)
    vram_bytes = peak_mem - initial_mem
    
    del tensor, _
    torch.cuda.empty_cache()
    
    return vram_bytes / (1024 * 1024)


@torch.no_grad()
def extract_feats(extractor,
                          video_paths: list,
                          save_dir: str,
                          patch_size: int = 32, # base_patch_size
                          resize_dim: tuple = (112, 112),
                          crop_dim: tuple = (112, 112),
                          target_fps: int = 30,
                          max_sec: float = 241.5,
                          safe_vram_mb: float = 3200.0,
                          skip_log_path: str = None):

    os.makedirs(save_dir, exist_ok=True)
    if skip_log_path is None:
        skip_log_path = os.path.join(save_dir, "skipped_videos.txt")

    num_gpus = torch.cuda.device_count()
    batch_size = max(1, num_gpus) 

    is_dp = isinstance(extractor, torch.nn.DataParallel)
    
    if num_gpus > 1 and not is_dp:
        extractor = torch.nn.DataParallel(extractor)
        is_dp = True

    core_model = extractor.module if is_dp else extractor
    clip_size = core_model.clip_size

    if not is_dp:
        extractor = extractor.to("cuda")
        
    extractor.eval()

    len_videos = len(video_paths)
    for video_idx, vp in enumerate(video_paths):
        video_features = []
        save_path = os.path.join(save_dir, os.path.basename(vp) + ".pt")

        if os.path.exists(save_path):
            continue

        try:
            cap_temp = cv2.VideoCapture(vp)
            fps_temp = cap_temp.get(cv2.CAP_PROP_FPS)
            frames_temp = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_temp.release()
            
            duration = frames_temp / fps_temp if fps_temp > 0 else 0
            patch_size_K = max(1.0, duration / max_sec)
            dynamic_patch_size = int(patch_size * patch_size_K)
            len_segments = dynamic_patch_size

            ratio = target_fps / fps_temp if fps_temp > 0 else 1.0
            target_frames = int(frames_temp * ratio)
            
            frame_per_segment = max(1, target_frames // len_segments)
            dummy = torch.rand(1, 3, int(frame_per_segment), *crop_dim)
            vram_usage_mb = calc_vram_usage_in_mb(extractor, dummy)
            
            if vram_usage_mb > safe_vram_mb:
                raise MemoryError(f"Only 1 segment wasting {vram_usage_mb:.2f} MB VRAM. (Limit: {safe_vram_mb} MB)")
            
            calculated_batch = int(safe_vram_mb // vram_usage_mb)
            if num_gpus > 1:
                batch_size = max(num_gpus, (calculated_batch // num_gpus) * num_gpus)
            else:
                batch_size = max(1, calculated_batch)

            segment_generator = fetch_video_patches(vp, target_fps=target_fps, patch_size=patch_size, 
                                                    resize_dim=resize_dim, crop_dim=crop_dim, max_sec=max_sec)
            
            if len_segments == 0:
                raise ValueError("No extracted segment in video")

            batch_buffer = []
            for segment_idx, segment in enumerate(segment_generator):
                segment = segment.float().div_(255.0)
                if segment.shape[2] < clip_size:
                    pad_size = clip_size - segment.shape[2]
                    segment = F.pad(segment, (0, 0, 0, 0, 0, pad_size))
                
                batch_buffer.append(segment)

                if len(batch_buffer) == batch_size or segment_idx == len_segments - 1:
                    
                    batch = torch.cat(batch_buffer, dim=0).to("cuda", non_blocking=True)
                    
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        feats = extractor(batch)  

                    video_features.append(feats.cpu()) 
                    batch_buffer.clear()

                progress_pct = ((segment_idx + 1) / len_segments) * 100
                info = (
                    f"Video: {video_idx+1:04d}/{len_videos} | "
                    f"Progress: %{progress_pct:5.2f} | "
                    f"Batch: {batch_size}"
                )
                
                print(info, end='\r', flush=True)

            print()

            video_feature = torch.cat(video_features, dim=0)  
            torch.save(video_feature, save_path)
            
        except Exception as e:
            with open(skip_log_path, "a") as f:
                f.write(f"{vp}\t{e}\n")
            print(f"\n[SKIP] {vp} -> {e}", flush=True)