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
            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 8),
            nn.Dropout(0.5),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        if self.training:
            noise = torch.rand_like(x) * 0.00025 - torch.rand_like(x) * 0.00025
            x = x + noise

        x = torch.nn.functional.normalize(x, dim=0)
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
        max_anomaly, _ = torch.max(y_anomaly, dim=1)
        max_normal, _ = torch.max(y_normal, dim=1)
        
        hinge_loss = F.relu(1.0 - max_anomaly + max_normal)
        
        smoothness = torch.sum((y_anomaly[:, :-1, :] - y_anomaly[:, 1:, :]) ** 2, dim=1)
        sparsity = torch.sum(y_anomaly, dim=1)
        avg_normal_loss = torch.mean(y_normal, dim=1)

        mean_hinge = torch.mean(hinge_loss)
        mean_smoothness = self.smoothness_K * torch.mean(smoothness)
        mean_sparsity = self.sparsity_K * torch.mean(sparsity)
        mean_avg_normal = self.normality_K * torch.mean(avg_normal_loss)
        
        return mean_hinge, mean_smoothness, mean_sparsity, mean_avg_normal

class MILVideoDataset(Dataset):
    def __init__(self, anomaly_files, normal_files, seq_len=32):
        self.anomaly_files = anomaly_files
        self.normal_files = normal_files
        self.seq_len = seq_len

    def __len__(self):
        return len(self.anomaly_files)

    def _interpolate_features(self, tensor):
        tensor = tensor.unsqueeze(0).permute(0, 2, 1)
        tensor = F.interpolate(tensor, size=self.seq_len, mode='linear', align_corners=False)
        tensor = tensor.squeeze(0).permute(1, 0)
        
        return tensor

    def __getitem__(self, idx):
        fa_path = self.anomaly_files[idx]
        fa = torch.load(fa_path, weights_only=True).float()
        fa = self._interpolate_features(fa)

        fn_path = random.choice(self.normal_files)
        fn = torch.load(fn_path, weights_only=True).float()
        fn = self._interpolate_features(fn)

        return fa, fn

def segment_score_model_trainer(model: SegmentRankingModel, 
                        anormal_feat_dir: str, 
                        normal_feat_dir: str, 
                        epochs: int=10, 
                        learning_rate: float=0.001,
                        k_fold: int=5,
                        batch_size: int=16,
                        val_ratio: float=0.2,
                        early_stop_patience: int=8,
                        pt_save_dir="segmentation_model_checkpoint"):

    anormal_files = [os.path.join(anormal_feat_dir, f) for f in os.listdir(anormal_feat_dir) if f.endswith('.pt')]
    normal_files = [os.path.join(normal_feat_dir, f) for f in os.listdir(normal_feat_dir) if f.endswith('.pt')]

    random.shuffle(anormal_files)
    random.shuffle(normal_files)

    def chunk_list(lst, k):
        n = len(lst)
        return [lst[i * n // k : (i + 1) * n // k] for i in range(k)]

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

        criterion = VideoSegmenterLoss(smoothness_K=0.001, sparsity_K=0.001, normality_K=0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0025)

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

            class_groups = defaultdict(list)
            for f in anormal_train_files:
                _class = re.match(r"^[a-zA-Z]+", os.path.basename(f)).group()
                class_groups[_class].append(f)
                
            class_names = list(class_groups.keys())
            class_pointers = {k: 0 for k in class_names}
            
            for k in class_names:
                random.shuffle(class_groups[k])
                
            anormal_batches = []
            class_idx = 0
            
            for i in range(num_batches):
                current_b_size = batch_size if i < num_batches - 1 else len(anormal_train_files) - i * batch_size
                if current_b_size == 0: current_b_size = batch_size
                
                batch = []
                for _ in range(current_b_size):
                    c = class_names[class_idx]
                    
                    if class_pointers[c] >= len(class_groups[c]):
                        random.shuffle(class_groups[c])
                        class_pointers[c] = 0
                        
                    batch.append(class_groups[c][class_pointers[c]])
                    class_pointers[c] += 1
                    
                    class_idx = (class_idx + 1) % len(class_names)
                    
                anormal_batches.append(batch)

            for i in range(num_batches):
                optimizer.zero_grad()

                anormal_batch_files = anormal_batches[i]
                current_b_size = len(anormal_batch_files)
                normal_batch_files = random.sample(normal_train_files, k=current_b_size)

                feat_anomaly = []
                for f in anormal_batch_files:
                    fa = torch.load(f, weights_only=True).to("cuda").float()
                    fa = fa.unsqueeze(0).permute(0, 2, 1)
                    fa = F.interpolate(fa, size=32, mode='linear', align_corners=False)
                    fa = fa.squeeze(0).permute(1, 0)
                    feat_anomaly.append(fa)
                    
                feat_normal = []
                for f in normal_batch_files:
                    fn = torch.load(f, weights_only=True).to("cuda").float()
                    fn = fn.unsqueeze(0).permute(0, 2, 1)
                    fn = F.interpolate(fn, size=32, mode='linear', align_corners=False)
                    fn = fn.squeeze(0).permute(1, 0)
                    feat_normal.append(fn)

                fa_batch = torch.stack(feat_anomaly)
                fn_batch = torch.stack(feat_normal)

                y_anomaly = model(fa_batch) 
                y_normal = model(fn_batch)
                
                h, sm, sp, an = criterion(y_anomaly, y_normal)
                
                train_loss = h + sm + sp + an
                train_loss.backward()
                
                batch_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.3)
                epoch_grad_norm += batch_grad_norm.item()
                
                optimizer.step()
                scheduler.step()

                epoch_mean_hinge_loss += h.item()
                epoch_mean_smoothness_loss += sm.item()
                epoch_mean_sparsity_loss += sp.item()
                epoch_avg_normal += an.item()

                train_losses.append(train_loss.item())

                progress_percent = ((i + 1) / num_batches) * 100
                print(f"Fold {fold+1} | Epoch {epoch_idx+1:03d}/{epochs} - Progress: %{progress_percent:5.2f}", end="\r")

            model.eval()
            val_mean_hinge, val_mean_smoothness, val_mean_sparsity, val_avg_normal = 0.0, 0.0, 0.0, 0.0
            
            with torch.no_grad():
                num_test_files = len(anormal_test_files)
                for fa_path, fn_path in zip(anormal_test_files, normal_test_files):
                    fa = torch.load(fa_path, weights_only=True).to("cuda").float()
                    fn = torch.load(fn_path, weights_only=True).to("cuda").float()
                    
                    y_anomaly = model(fa.unsqueeze(0))
                    y_normal = model(fn.unsqueeze(0))
                    
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

def compute_tiou(pred_segment, gt_segment):
    intersection = max(0, min(pred_segment[1], gt_segment[1]) - max(pred_segment[0], gt_segment[0]))
    union = (pred_segment[1] - pred_segment[0]) + (gt_segment[1] - gt_segment[0]) - intersection
    
    if union > 0:
        return intersection / union
    return 0.0

def calculate_mAP(datas, tiou_threshold=0.5):
    def _get_ap(subset_datas):
        all_preds = []
        total_gts = 0
        gt_dict = {}

        for data in subset_datas:
            video_name = data['name']
            gts = data['gt_time_stamps']
            preds = data['pred_time_stamps']

            total_gts += len(gts)
            gt_dict[video_name] = [{'interval': gt, 'matched': False} for gt in gts]

            for pred in preds:
                all_preds.append({
                    'video_name': video_name,
                    'interval': [pred[0], pred[1]],
                    'score': pred[2]
                })

        if total_gts == 0: 
            return 0.0

        all_preds = sorted(all_preds, key=lambda x: x['score'], reverse=True)

        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))

        for idx, pred in enumerate(all_preds):
            video_name = pred['video_name']
            pred_interval = pred['interval']
            
            video_gts = gt_dict.get(video_name, [])
            
            best_tiou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(video_gts):
                tiou = compute_tiou(pred_interval, gt['interval'])
                if tiou > best_tiou:
                    best_tiou = tiou
                    best_gt_idx = gt_idx

            if best_tiou >= tiou_threshold and not video_gts[best_gt_idx]['matched']:
                tp[idx] = 1
                video_gts[best_gt_idx]['matched'] = True
            else:
                fp[idx] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recalls = cum_tp / total_gts
        precisions = cum_tp / (cum_tp + cum_fp)

        mrec = np.concatenate(([0.], recalls, [1.]))
        mpre = np.concatenate(([0.], precisions, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
        return ap

    # 1. Genel (Overall) mAP hesaplaması
    overall_ap = _get_ap(datas)

    # 2. Kategori bazlı AP hesaplaması
    categories = sorted(list(set(data['category'] for data in datas if data['category'] != 'Normal')))
    category_aps = {}
    
    for cat in categories:
        # Sadece o kategoriye ait verileri filtrele
        cat_datas = [d for d in datas if d['category'] == cat]
        category_aps[cat] = _get_ap(cat_datas)
        
    return overall_ap, category_aps