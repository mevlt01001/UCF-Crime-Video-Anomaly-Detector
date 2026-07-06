import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .C3D_model import C3D_FeatureExtractor
from .video_preprocess import fetch_video_patches

class VideoSegmenter(nn.Module):
    def __init__(self):
        super(VideoSegmenter, self).__init__()
        self.feature_extractor = C3D_FeatureExtractor()
        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        features = self.feature_extractor(x)
        x = self.relu(self.fc1(features))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x
    
if __name__ == "__main__":
    model = VideoSegmenter().to("cuda")
    data = torch.randn(2, 3, 13, 240, 240).to("cuda")
    output = model(data)
    print(output)

from torch.utils.data import Dataset, DataLoader

class VideoPatchesDataset(Dataset):
    def __init__(self, video_path:os.PathLike, target_fps:int=30, patch_size:int=32, resize_dim:tuple=(320, 320)):
        self.videos_path = video_path
        self.patches = [patch for patch in fetch_video_patches(video_path, target_fps, patch_size, resize_dim)]

        print(f"VideoPatchesDataset: {len(self.patches)} first.shape: {self.patches[0].shape if self.patches else 'N/A'}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

class VideoSegmenterLoss(nn.Module):
    def __init__(self, lambda_1=8e-5, lambda_2=8e-5):

        super(VideoSegmenterLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    @torch.no_grad()
    def forward(self, y_anomaly, y_normal):
        """
        y_anomaly: (Batch, Num_Segments)
        y_normal: (Batch, Num_Segments)
        """

        y_anomaly = y_anomaly
        y_normal = y_normal

        max_anomaly, _ = torch.max(y_anomaly, dim=1)  # Boyut: (Batch,)
        max_normal, _ = torch.max(y_normal, dim=1)    # Boyut: (Batch,)
        
        hinge_loss = F.relu(1.0 - max_anomaly + max_normal)
        smoothness = torch.sum((y_anomaly[:, :-1] - y_anomaly[:, 1:]) ** 2, dim=1)
        sparsity = torch.sum(y_anomaly, dim=1)
        
        mean_hinge = torch.mean(hinge_loss)
        mean_smoothness = self.lambda_1 * torch.mean(smoothness)
        mean_sparsity = self.lambda_2 * torch.mean(sparsity)
        
        total_loss = mean_hinge + mean_smoothness + mean_sparsity
        
        return total_loss

def video_segmenter_trainer(model: VideoSegmenter, 
                          normal_videos_path:os.PathLike, 
                          anormal_videos_path:os.PathLike, 
                          resize_dim:tuple=(320, 320),
                          epochs:int=10, 
                          patch_size:int=32,
                          batch_size:int=8,
                          learning_rate:float=0.001):
    
    criterion = VideoSegmenterLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    normal_video_paths = normal_videos_path
    anormal_video_paths = anormal_videos_path

    best_loss = float('inf')
    
    for epoch_idx in range(epochs):

        random.shuffle(normal_video_paths)
        random.shuffle(anormal_video_paths)

        epoch_loss = 0.0
        epoch_steps = 0

        for anormal_video_idx, anormal_video_path in enumerate(anormal_video_paths):
            batch_loss = 0.0
            batch_steps = 0

            normal_video_path = random.choice(normal_video_paths)

            anormal_dataset = VideoPatchesDataset(anormal_video_path, patch_size=patch_size, resize_dim=resize_dim)
            normal_dataset = VideoPatchesDataset(normal_video_path, patch_size=patch_size, resize_dim=resize_dim)
            
            anormal_loader = DataLoader(anormal_dataset, batch_size=batch_size, shuffle=True)
            normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True)
            
            optimizer.zero_grad()
            
            for (anormal_batch, normal_batch) in zip(anormal_loader, normal_loader):

                anormal_batch = anormal_batch.squeeze(1).to("cuda")
                normal_batch = normal_batch.squeeze(1).to("cuda")

                print(f"anormal_batch.shape: {anormal_batch.shape}, normal_batch.shape: {normal_batch.shape}")

                y_anomaly = model.forward(anormal_batch)
                y_normal = model.forward(normal_batch)
            
                loss += criterion.forward(y_anomaly, y_normal)
                batch_steps += 1

                print(f"Epoch {epoch_idx+1:03d}/{epochs} - Processing {(anormal_video_idx+1)/len(anormal_video_paths)*100:05.2f}% Loss: {loss.item()/batch_steps:.6f}", end="\r")

            loss /= batch_steps
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
                
            