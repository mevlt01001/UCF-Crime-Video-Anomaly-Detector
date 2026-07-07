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
    def __init__(self, input_dim=4096):
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

def MIL_network_trainer(model: MILRankingNetwork, 
                        anormal_feat_dir: str, 
                        normal_feat_dir: str, 
                        epochs: int=10, 
                        learning_rate: float=0.001,
                        test_ratio:int=0.2,
                        batch_size: int=16):

    anormal_files = [os.path.join(anormal_feat_dir, f) for f in os.listdir(anormal_feat_dir) if f.endswith('.pt')]
    normal_files = [os.path.join(normal_feat_dir, f) for f in os.listdir(normal_feat_dir) if f.endswith('.pt')]

    anormal_test_files = anormal_files[0:int(len(anormal_files)*test_ratio)]
    normal_test_files = normal_files[0:int(len(normal_files)*test_ratio)]

    anormal_train_files = anormal_files[len(anormal_test_files):]
    normal_train_files = normal_files[len(normal_test_files):]


    model = model.train(True)
    criterion = VideoSegmenterLoss(lambda_1=8e-5, lambda_2=8e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs*(len(anormal_train_files) + batch_size - 1) // batch_size, eta_min=0.0005)

    best_loss = float("inf")

    for epoch_idx in range(epochs):
        random.shuffle(anormal_files)

        epoch_loss = 0.0
        num_batches = (len(anormal_train_files) + batch_size - 1) // batch_size

        for i in range(num_batches):
            optimizer.zero_grad()

            anormal_batch_files = anormal_train_files[i * batch_size : (i + 1) * batch_size]
            current_b_size = len(anormal_batch_files)

            normal_batch_files = random.choices(normal_train_files, k=current_b_size)

            feat_anomaly = torch.stack([torch.load(f) for f in anormal_batch_files]).to("cuda")
            feat_normal = torch.stack([torch.load(f) for f in normal_batch_files]).to("cuda")
            
            y_anomaly = model.forward(feat_anomaly) # [B, 32]
            y_normal = model.forward(feat_normal)   # [B, 32]
        
            loss = criterion.forward(y_anomaly, y_normal)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            
            progress_percent = ((i + 1) / num_batches) * 100
            print(f"Epoch {epoch_idx+1:03d}/{epochs} - Progress: %{progress_percent:5.3f} - Loss: {loss.item():.6f}", end="\r")

        with torch.no_grad():

            normal_batch_files = random.choices(normal_test_files, k=len(anormal_test_files))

            feat_anomaly = torch.stack([torch.load(f) for f in anormal_test_files]).to("cuda")
            feat_normal = torch.stack([torch.load(f) for f in normal_batch_files]).to("cuda")
            
            y_anomaly = model.forward(feat_anomaly) # [B, 32]
            y_normal = model.forward(feat_normal)   # [B, 32]
            val_loss = criterion.forward(y_anomaly, y_normal).item()
        
        new_best_loss = val_loss < best_loss
        print(f"Epoch {epoch_idx+1:03d}/{epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}- Avg. Loss: {epoch_loss/num_batches:.6f} - Val. Loss: {val_loss:.6f} {'*' if new_best_loss else ''}", end="\n")
        
        if new_best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best.pt")

        torch.save(model.state_dict(), "last.pt")