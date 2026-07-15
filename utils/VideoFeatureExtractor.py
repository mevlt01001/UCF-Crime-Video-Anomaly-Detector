import os
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .video_preprocess import fetch_video_patches
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class FeatureExtractor(nn.Module):
    def __init__(self, clip_size: int = 16, overlap: int = 0):
        super(FeatureExtractor, self).__init__()
        self.clip_size = clip_size
        self.overlap = overlap
        self.stride = clip_size - overlap

        weights = R2Plus1D_18_Weights.DEFAULT
        self.backbone = r2plus1d_18(weights=weights)
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, S, H, W = x.shape

        pad_size = max(0, self.clip_size - S)
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_size))
            S = x.shape[2]

        x_unfolded = x.unfold(2, self.clip_size, self.stride)
        x_clips = x_unfolded.permute(0, 2, 1, 5, 3, 4)
        num_clips = x_clips.shape[1]

        h = x_clips.reshape(-1, C, self.clip_size, H, W)    # [B, 3, S, H, W]
        h = self.backbone(h)
        h = F.normalize(h, p=2, dim=-1)
        h = h.reshape(B, num_clips, -1)
        h = h.mean(dim=1)   # [B, 512]

        return h
    
    @torch.no_grad()
    def extract_feats(self,
                      video_path: os.PathLike, 
                      patch_size: int, 
                      fps: int, 
                      resolution: tuple[int, int],
                      safe_vram_mb: int = 3200):
        
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated()
        dummy = torch.randn(1, 3, self.clip_size, *resolution, device="cuda", dtype=torch.float32)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16): _ = self.forward(dummy)
        peak_mem = torch.cuda.max_memory_allocated()
        bytes_per_clip = peak_mem - initial_mem
        
        del dummy, _
        torch.cuda.empty_cache()
        
        length_of_clip = _probe_segment_length(video_path, fps, patch_size)
        number_of_clip = _num_clips_for_segment(length_of_clip, self.clip_size, self.stride)
        bytes_per_segment = bytes_per_clip * number_of_clip
        
        safe_vram_bytes = safe_vram_mb * 1024 * 1024
        batch_size = max(1, int(safe_vram_bytes // bytes_per_segment))

        patches = [patch for patch in fetch_video_patches(video_path, fps, patch_size, resolution)]

        feats = []

        with tqdm(total=patch_size, desc="Processed Patches") as pbar:

            for start_idx in range(0, patch_size, batch_size):
                end_idx = min(patch_size, start_idx + batch_size)

                batch = torch.concat(patches[start_idx:end_idx], dim=0)
                batch = batch.to("cuda", non_blocking=True).float().div_(255.0)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    feat = self.forward(batch)

                feats.append(feat.detach().float())
                pbar.update(end_idx-start_idx)

                del batch, feat
                torch.cuda.empty_cache()
            
            return torch.concat(feats, dim=0)


def _probe_segment_length(video_path: str, target_fps: int, patch_size: int):
    """
    This method calculates and retunrs the length of segmments of patches.
    """

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
    S = target_frames // patch_size
    return S

def _num_clips_for_segment(S: int, clip_size: int, stride: int) -> int:
    """
    Calculates and returns number of clips for segments.
    """
    S_eff = max(S, clip_size) 
    return (S_eff - clip_size) // stride + 1

def _number_of_patch_bytes(num_clips_in_batch: int, resize_dim: tuple,
                           clip_size: int, dtype_bytes: int = 4,
                           channels: int = 64) -> int:
    """
    Calculates and returns each patch's bytes
    """
    H, W = resize_dim
    bytes_per_clip = channels * clip_size * H * W * dtype_bytes
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
    """
    Extracts features from given video_paths and saves the features to given save_dir.
    """

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
    bytes_per_clip = _number_of_patch_bytes(1, resize_dim, clip_size)

    skipped = []

    for vp in tqdm(video_paths):
        save_path = os.path.join(save_dir, os.path.basename(vp) + ".pt")
        if os.path.exists(save_path):
            continue

        S = _probe_segment_length(vp, target_fps, patch_size)
        if S is None:
            print(f"[SKIP] {vp}: Metadata did not read!")
            skipped.append((vp, "Metadata did not read!"))
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
            print(f"\n[SKIP] {vp}: Expected {patch_size} segment, got {num_segments_seen}")
            torch.cuda.empty_cache()
            continue

        video_feature = torch.cat(video_feature_list, dim=0)  # (patch_size, 4096)
        torch.save(video_feature, save_path)

        del video_feature_list, video_feature
        torch.cuda.empty_cache()

    if skipped:
        with open(skip_log_path, "w") as f:
            for vp, reason in skipped:
                f.write(f"{vp}\t{reason}\n")
        print(f"\nNumber of {len(skipped)} videos skipped due to VRAM -> {skip_log_path}")

if __name__ == "__main__":
    data = torch.randn(1, 3, 54, 112, 112).to("cuda")
    model = FeatureExtractor().to("cuda")
    output = model(data)
    print(output.shape)