import torch
import torch.nn as nn
from C3D_model import C3D_FeatureExtractor

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