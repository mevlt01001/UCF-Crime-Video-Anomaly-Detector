import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .video_preprocess import fetch_video_patches

class MILRankingNetwork(nn.Module):
    def __init__(self, input_dim=8192):
        super(MILRankingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        # x.shape: (Batch, 32, 8192)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x.squeeze(-1) # (Batch, 32)

class VideoSegmenterLoss(nn.Module):
    def __init__(self, lambda_1=8e-5, lambda_2=8e-5):
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


@torch.no_grad()
def extract_C3D_features(extractor,
                          video_paths: list,
                          save_dir: str,
                          patch_size: int = 32,
                          batch: int = 4,
                          resize_dim: tuple = (112, 112)):

    os.makedirs(save_dir, exist_ok=True)

    is_dp = isinstance(extractor, torch.nn.DataParallel)
    if not is_dp:
        extractor = extractor.to("cuda")
    extractor.eval()

    for vp in tqdm(video_paths):
        save_path = os.path.join(save_dir, os.path.basename(vp) + ".pt")
        if os.path.exists(save_path):
            continue

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

        for segment in fetch_video_patches(vp, target_fps=30, patch_size=patch_size, resize_dim=resize_dim):
            # segment: (1, 3, seq_len, H, W) uint8
            mini_batch_buffer.append(segment)
            num_segments_seen += 1

            if len(mini_batch_buffer) == batch:
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


def MIL_network_trainer(model: MILRankingNetwork, 
                        anormal_feat_dir: str, 
                        normal_feat_dir: str, 
                        epochs: int=10, 
                        learning_rate: float=0.001):
    
    criterion = VideoSegmenterLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    anormal_files = [os.path.join(anormal_feat_dir, f) for f in os.listdir(anormal_feat_dir) if f.endswith('.pt')]
    normal_files = [os.path.join(normal_feat_dir, f) for f in os.listdir(normal_feat_dir) if f.endswith('.pt')]

    best_loss = float("inf")  

    for epoch_idx in range(epochs):
        random.shuffle(anormal_files)
        random.shuffle(normal_files)

        epoch_loss = 0.0

        for i, anormal_file in enumerate(anormal_files):
            normal_file = random.choice(normal_files)

            feat_anomaly = torch.load(anormal_file).unsqueeze(0).to("cuda") # (1, 32, 8192)
            feat_normal = torch.load(normal_file).unsqueeze(0).to("cuda")   # (1, 32, 8192)

            optimizer.zero_grad()
            
            y_anomaly = model.forward(feat_anomaly) # [1, 32]
            y_normal = model.forward(feat_normal)   # [1, 32]
        
            loss = criterion.forward(y_anomaly, y_normal)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            print(f"Epoch {epoch_idx+1:03d}/{epochs} - Progress: {(i+1)/len(anormal_files):5.3f} - Loss: {loss.item():.6f}", end="\r")
        
        avg_loss = epoch_loss / len(anormal_files)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best.pt")

        torch.save(model.state_dict(), "last.pt")