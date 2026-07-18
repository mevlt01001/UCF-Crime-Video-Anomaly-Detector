import random
import torch, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime
from .video_preprocess import fetch_video_patches, get_report_dir
from .visualization_tools import plot_anomaly_timeline

class SegmentRankingModel(nn.Module):
    def __init__(self, input_dim=512):
        super(SegmentRankingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        x = self.fc1.forward(x)
        
        x = self.dropout.forward(x)

        x = self.fc2.forward(x)
        x = self.dropout.forward(x)

        x = self.fc3.forward(x)
        x = self.relu.forward(x)
        x = self.dropout.forward(x)

        x = self.fc4.forward(x)
        x = self.sigmoid.forward(x)
        return x
    
    @torch.no_grad()
    def score_to_segments(self, 
                   patch_feats:torch.Tensor, 
                   video_seconds:float, 
                   threshold:float=0.3, 
                   tolerance_sec:float=3.0, 
                   padding_sec:float=3.0,
                   plot_graph:bool=False,
                   save_file_name: str = "anomaly_segmentation_plot.png"
                   ):
        """
        VideoFeatureExtractor çıktısı olan özellik vektörlerini alarak
        bu özellik vektörlerini skorlar ve skorlara göre anormal kısımların segmetasyonu yapılır.

        Segmentasyon yapılırken `threshold` değerinin altındaki tüm patch'ler anormal olmayan patch'lar
        olarak nitelendirilirken iki anormal segment arası `tolerance_sec` saniyeden daha az bir süre var ise
        anormal segmentasyonlar birleştirilier. Her anormal segmnetsayon öncesi ve sonrası `padding_sec` kadar video
        saniye segmentasyona dahil edilir.

        Args:
            patch_feats (torch.Tensor): Extracted features for each patch from FeatrueExtractor model
            video_seconds (int): Number of seconds which extracted video from
            threshold (float): Abnormal event threshold
            tolerance_sec (float): A tolerance to segmentate Abnormal event from normal event
            padding_sec(float): Segmentation padding before and after abnormal event
        """ 
        
        scores_org = self.forward(patch_feats).squeeze(-1) # [32]
        scores_org = scores_org.unsqueeze(0).unsqueeze(0)  # [1,1,32]
        
        kernel_size = 21
        scores_linear_interpolate  = F.interpolate(scores_org, size=1000, mode='linear', align_corners=True)
        scores_nearest_interpolate = F.interpolate(scores_org, size=1000, mode='nearest')
        scores_linear_interpolate  = F.pad(scores_linear_interpolate, (kernel_size // 2, kernel_size // 2), mode='reflect')
        scores_linear_interpolate  = F.avg_pool1d(scores_linear_interpolate, kernel_size=kernel_size, stride=1)
        
        scores_linear_interpolate  = scores_linear_interpolate.squeeze(0).squeeze(0)
        scores_nearest_interpolate = scores_nearest_interpolate.squeeze(0).squeeze(0)
        
        dt = video_seconds / 1000
        anomaly_indices = torch.where(scores_linear_interpolate >= threshold)[0].tolist()
        
        final_segments = []
        
        if anomaly_indices:
            raw_segments = []
            current_start = anomaly_indices[0]
            current_end = anomaly_indices[0]

            for idx in anomaly_indices[1:]:
                time_gap = (idx - current_end) * dt

                if time_gap <= tolerance_sec:
                    current_end = idx
                else:
                    raw_segments.append((current_start, current_end))
                    current_start = idx
                    current_end = idx

            raw_segments.append((current_start, current_end))
            
            for start_idx, end_idx in raw_segments:
                start_time = start_idx * dt
                end_time = min(video_seconds, end_idx * dt + padding_sec)

                padded_start = max(0.0, start_time - padding_sec)

                final_segments.append({
                    "start_time": round(padded_start, 2),
                    "end_time": round(end_time, 2),
                    "duration": round(end_time - padded_start, 2)
                })

        if plot_graph:
            save_root = get_report_dir(save_file_name)
            os.makedirs(save_root, exist_ok=True)
            plot_anomaly_timeline(scores_linear_interpolate.cpu(),
                                  scores_nearest_interpolate.cpu(),
                                  final_segments,
                                  video_seconds,
                                  threshold,
                                  os.path.join(save_root, "abnormal_segments_graph.png"))

        return final_segments
    


    
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

def segment_score_model_trainer(model: SegmentRankingModel,
                        anormal_feat_dir: str, 
                        normal_feat_dir: str, 
                        epochs: int=10, 
                        learning_rate: float=0.001,
                        test_ratio: float=0.2,
                        batch_size: int=16,
                        pt_save_dir = "segmentation_model_checkpoint"):

    anormal_files = [os.path.join(anormal_feat_dir, f) for f in os.listdir(anormal_feat_dir) if f.endswith('.pt')]
    normal_files = [os.path.join(normal_feat_dir, f) for f in os.listdir(normal_feat_dir) if f.endswith('.pt')]

    random.shuffle(anormal_files)
    random.shuffle(normal_files)

    num_anormal_test = int(len(anormal_files) * test_ratio)
    anormal_test_files = anormal_files[:num_anormal_test]
    anormal_train_files = anormal_files[num_anormal_test:]

    num_normal_test = int(len(normal_files) * test_ratio)
    normal_test_files = normal_files[:num_normal_test]
    normal_train_files = normal_files[num_normal_test:]

    if len(normal_test_files) >= len(anormal_test_files):
        static_val_normal_files = random.sample(normal_test_files, k=len(anormal_test_files))
    else:
        static_val_normal_files = random.choices(normal_test_files, k=len(anormal_test_files))

    criterion = VideoSegmenterLoss(lambda_1=1e-4, lambda_2=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    
    num_batches = (len(anormal_train_files) + batch_size - 1) // batch_size
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        epochs * num_batches, 
        eta_min=0.00005
    )

    best_loss    = float("inf")
    train_losses = []
    val_losses   = []

    for epoch_idx in range(epochs):
        model.train()

        random.shuffle(anormal_train_files)

        epoch_loss = 0.0

        for i in range(num_batches):
            optimizer.zero_grad()

            anormal_batch_files = anormal_train_files[i * batch_size : (i + 1) * batch_size]
            current_b_size = len(anormal_batch_files)

            normal_batch_files = random.sample(normal_train_files, k=current_b_size)

            feat_anomaly = torch.stack([torch.load(f) for f in anormal_batch_files]).to("cuda")
            feat_normal = torch.stack([torch.load(f) for f in normal_batch_files]).to("cuda")
            
            y_anomaly = model.forward(feat_anomaly) # [B, 32]
            y_normal = model.forward(feat_normal)   # [B, 32]
        
            train_loss = criterion.forward(y_anomaly, y_normal)
            
            train_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()

            epoch_loss += train_loss.item()
            train_losses.append(train_loss.item())

            progress_percent = ((i + 1) / num_batches) * 100
            print(f"Epoch {epoch_idx+1:03d}/{epochs} - Progress: %{progress_percent:5.3f} - Grad Norm: {grad_norm.item():6.3f} - Loss: {train_loss.item():.6f}", end="\r")

        model.eval()
        with torch.no_grad():
            feat_anomaly = torch.stack([torch.load(f) for f in anormal_test_files]).to("cuda")
            feat_normal = torch.stack([torch.load(f) for f in static_val_normal_files]).to("cuda")
            
            y_anomaly = model.forward(feat_anomaly) # [B, 32]
            y_normal = model.forward(feat_normal)   # [B, 32]
            val_loss = criterion.forward(y_anomaly, y_normal).item()
            val_losses.append(val_loss)
        
        new_best_loss = val_loss < best_loss 
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Epoch {epoch_idx+1:03d}/{epochs} - "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} - "
              f"Avg. Loss: {epoch_loss/num_batches:.6f} - "
              f"Val. Loss: {val_loss:.6f} {'*' if new_best_loss else ''}", 
              end="\n")
        
        os.makedirs(pt_save_dir, exist_ok=True)

        checkpoint = {
            "validation_loss": val_loss,
            "state_dict": model.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses
        }

        if new_best_loss:
            best_loss = val_loss
            torch.save(checkpoint, f"{pt_save_dir}/best_loss.pt")

        torch.save(checkpoint, f"{pt_save_dir}/last.pt")