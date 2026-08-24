import copy
import random
import torch, os, re
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from .video_preprocess import fetch_video_patches, get_report_dir
from .visualization_tools import plot_anomaly_timeline

class SegmentRankingModel(nn.Module):
    def __init__(self, input_dim=1024):
        super(SegmentRankingModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.7),

            nn.Linear(512, 256),
            nn.Dropout(0.6),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        if self.training:
            noise = torch.rand_like(x) * 0.05 - torch.rand_like(x) * 0.05
            x = x + noise

        x = torch.nn.functional.normalize(x, dim=1)
        x = self.mlp(x)
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
    def __init__(self, smoothness_K=0, sparsity_K=0, normality_K=0):
        super(VideoSegmenterLoss, self).__init__()
        self.smoothness_K = smoothness_K
        self.sparsity_K = sparsity_K
        self.normality_K = normality_K

    def forward(self, y_anomaly, y_normal):
        # print(f"y_anomaly.shape: {y_anomaly.shape}, y_normal.shape: {y_normal.shape}"), exit()
        max_anomaly, _ = torch.max(y_anomaly, dim=0)
        max_normal, _ = torch.max(y_normal, dim=0)
        
        hinge_loss = F.relu(1.0 - max_anomaly + max_normal)
        
        smoothness = torch.sum((y_anomaly[:-1, :] - y_anomaly[1:, :]) ** 2, dim=0)
        sparsity = torch.sum(y_anomaly, dim=0)
        avg_normal_loss = torch.mean(y_normal, dim=0)

        mean_hinge = torch.mean(hinge_loss)
        mean_smoothness = self.smoothness_K * torch.mean(smoothness)
        mean_sparsity = self.sparsity_K * torch.mean(sparsity)
        mean_avg_normal = self.normality_K * torch.mean(avg_normal_loss)
        
        return mean_hinge, mean_smoothness, mean_sparsity, mean_avg_normal

import os
import copy
import random
import re
from datetime import datetime
from collections import defaultdict
import torch

def segment_score_model_trainer(model: SegmentRankingModel, 
                        anormal_feat_dir: str, 
                        normal_feat_dir: str, 
                        epochs: int=10, 
                        learning_rate: float=0.001,
                        k_fold: int=5,
                        batch_size: int=16,
                        val_ratio: float=0.2,
                        early_stop_patience: int=8,
                        weight_decay:float = 0.001,
                        Normal_loss_k:float = 0.0001,
                        Sparsity_loss_k:float = 0.0001,
                        Smoothness_loss_k:float = 0.0001,
                        pt_save_dir="segmentation_model_checkpoint"):

    anormal_files = [os.path.join(anormal_feat_dir, f) for f in os.listdir(anormal_feat_dir) if f.endswith('.pt')]
    normal_files = [os.path.join(normal_feat_dir, f) for f in os.listdir(normal_feat_dir) if f.endswith('.pt')]

    random.shuffle(anormal_files)
    random.shuffle(normal_files)

    def chunk_list(file_lst, k):
        n = len(file_lst)
        file_count_per_chunk = n // k
        return [
                   file_lst[i * file_count_per_chunk : (i + 1) * file_count_per_chunk] 
                   for i in range(k)
               ]

    anormal_k_chunks = chunk_list(anormal_files, k_fold)

    initial_model_weights = copy.deepcopy(model.state_dict())
    
    os.makedirs(pt_save_dir, exist_ok=True)
    fold_best_losses = []

    for fold in range(k_fold):
        print(f"\n{'='*20} FOLD {fold+1}/{k_fold} {'='*20}")
        
        model.load_state_dict(initial_model_weights)
        model.to("cuda")

        if k_fold == 1:
            n_anormal_val = max(1, int(len(anormal_files) * val_ratio))
            anormal_test_files = anormal_files[:n_anormal_val]
            anormal_train_files = anormal_files[n_anormal_val:]
        else:
            anormal_test_files = anormal_k_chunks[fold]
            anormal_train_files = [f for i, chunk in enumerate(anormal_k_chunks) if i != fold for f in chunk]

        normal_train_files = normal_files 

        if len(normal_files) < len(anormal_test_files):
            normal_test_files = random.choices(normal_files, k=len(anormal_test_files))
        else:
            normal_test_files = random.sample(normal_files, k=len(anormal_test_files))

        criterion = VideoSegmenterLoss(Smoothness_loss_k, Sparsity_loss_k, Normal_loss_k)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        num_batches = (len(anormal_train_files) + batch_size - 1) // batch_size
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs * num_batches, 
            eta_min=0.00005
        )
        
        best_loss = float("inf")
        patience_counter = 0 
        train_losses = []
        val_losses = []

        for epoch_idx in range(epochs):
            model.train()
            epoch_mean_hinge_loss = 0.0
            epoch_mean_smoothness_loss = 0.0
            epoch_mean_sparsity_loss = 0.0
            epoch_avg_normal = 0.0
            epoch_grad_norm = 0.0 

            random.shuffle(anormal_train_files)
            anormal_batches = [
                anormal_train_files[i * batch_size : (i + 1) * batch_size] 
                for i in range(num_batches)
            ]

            for i in range(num_batches):
                optimizer.zero_grad()

                anormal_batch_files = anormal_batches[i]
                current_b_size = len(anormal_batch_files)
                
                normal_batch_files = random.sample(normal_train_files, k=current_b_size)

                batch_h, batch_sm, batch_sp, batch_an = 0.0, 0.0, 0.0, 0.0
                
                for fa_path, fn_path in zip(anormal_batch_files, normal_batch_files):
                    fa = torch.load(fa_path, weights_only=True)["feats"].to("cuda").float()
                    fn = torch.load(fn_path, weights_only=True)["feats"].to("cuda").float()

                    y_anomaly = model(fa) 
                    y_normal = model(fn)
                    
                    h, sm, sp, an = criterion(y_anomaly, y_normal)
                    
                    batch_h += h
                    batch_sm += sm
                    batch_sp += sp
                    batch_an += an
                    
                batch_h /= current_b_size
                batch_sm /= current_b_size
                batch_sp /= current_b_size
                batch_an /= current_b_size
                
                train_loss = batch_h + batch_sm + batch_sp + batch_an
                train_loss.backward()
                
                batch_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.75)
                epoch_grad_norm += batch_grad_norm.item()
                
                optimizer.step()
                scheduler.step()

                epoch_mean_hinge_loss += batch_h.item()
                epoch_mean_smoothness_loss += batch_sm.item()
                epoch_mean_sparsity_loss += batch_sp.item()
                epoch_avg_normal += batch_an.item()

                train_losses.append(train_loss.item())

                progress_percent = ((i + 1) / num_batches) * 100
                print(f"Fold {fold+1} | Epoch {epoch_idx+1:03d}/{epochs} - Progress: %{progress_percent:5.2f}", end="\r")

            model.eval()
            val_mean_hinge, val_mean_smoothness, val_mean_sparsity, val_avg_normal = 0.0, 0.0, 0.0, 0.0
            
            with torch.no_grad():
                num_test_files = len(anormal_test_files)
                for fa_path, fn_path in zip(anormal_test_files, normal_test_files):
                    fa = torch.load(fa_path, weights_only=True)["feats"].to("cuda").float()
                    fn = torch.load(fn_path, weights_only=True)["feats"].to("cuda").float()
                    
                    y_anomaly = model(fa)
                    y_normal = model(fn)
                    
                    h, sm, sp, an = criterion(y_anomaly, y_normal)
                    val_mean_hinge += h.item()
                    val_mean_smoothness += sm.item()
                    val_mean_sparsity += sp.item()
                    val_avg_normal += an.item()
                
                val_mean_hinge /= num_test_files
                val_mean_smoothness /= num_test_files
                val_mean_sparsity /= num_test_files
                val_avg_normal /= num_test_files
                
                val_loss = val_mean_hinge + val_mean_smoothness + val_mean_sparsity + val_avg_normal
                val_losses.append(val_loss)
            
            new_best_loss = val_loss < best_loss
            
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Fold {fold+1} | Epoch {epoch_idx+1:03d}/{epochs} - "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}\n"
                  f"Train Cls Loss: {epoch_mean_hinge_loss/num_batches:.6f}\n"
                  f"Train Normal Loss: {epoch_avg_normal/num_batches:.6f}\n"
                  f"Train Sparsity Loss: {epoch_mean_sparsity_loss/num_batches:.6f}\n"
                  f"Train Smoothness Loss: {epoch_mean_smoothness_loss/num_batches:.6f}\n"
                  f"Train Avg Grad Norm: {epoch_grad_norm/num_batches:.6f}\n"
                  f"Val Cls Loss: {val_mean_hinge:.6f}\n"
                  f"Val Normal Loss: {val_avg_normal:.6f}\n"
                  f"Val Sparsity Loss: {val_mean_sparsity:.6f}\n"
                  f"Val Smoothness Loss: {val_mean_smoothness:.6f}\n"
                  f"Val. Loss: {val_loss:.6f} {'*' if new_best_loss else ''}\n")
            
            checkpoint = {
                "fold": fold + 1,
                "epoch": epoch_idx + 1,
                "validation_loss": val_loss,
                "state_dict": model.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses
            }

            if new_best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(checkpoint, os.path.join(pt_save_dir, f"best_loss_fold_{fold+1}.pt"))
            else:
                patience_counter += 1

            torch.save(checkpoint, os.path.join(pt_save_dir, f"last_fold_{fold+1}.pt"))

            if patience_counter >= early_stop_patience:
                print(f"Early stopping fold {fold+1} at epoch {epoch_idx+1} "
                      f"(no val improvement for {early_stop_patience} epochs)")
                break
        
        fold_best_losses.append(best_loss)
        print(f"--- Fold {fold+1} Completed. Best Val Loss: {best_loss:.6f}")

    print("\nK-Fold Cross Validation Completed!")
    print(f"{k_fold} Fold Avg. Val Loss: {sum(fold_best_losses) / k_fold:.6f}")