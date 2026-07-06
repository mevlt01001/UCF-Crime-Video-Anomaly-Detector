import torch
import torch.nn as nn
import torch.nn.functional as F

class C3D_FeatureExtractor(nn.Module):

    def __init__(self, clip_size: int = 16, overlap: int = 8):
        super(C3D_FeatureExtractor, self).__init__()
        self.clip_size = clip_size
        self.overlap = overlap
        self.stride = clip_size - overlap

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape: (B, C, S, H, W)
        """
        B, C, S, H, W = x.shape

        if S < self.clip_size:
            pad_size = self.clip_size - S
            x = F.pad(x, (0, 0, 0, 0, 0, pad_size))
            S = self.clip_size

        clips = []
        for start_idx in range(0, S - self.clip_size + 1, self.stride):
            clips.append(x[:, :, start_idx : start_idx + self.clip_size, :, :])
            
        last_start = list(range(0, S - self.clip_size + 1, self.stride))[-1]
        if last_start + self.clip_size < S:
            clips.append(x[:, :, S - self.clip_size : S, :, :])

        x_clips = torch.stack(clips, dim=1)
        num_clips = x_clips.shape[1]

        h = x_clips.view(B * num_clips, C, self.clip_size, H, W)

        h = self.relu(self.conv1(h))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(B, num_clips, -1)
        
        h = h.mean(dim=1)

        return h

if __name__ == "__main__":

    data = torch.randn(2, 3, 13, 128, 128).to("cuda")
    model = C3D_FeatureExtractor().to("cuda")
    output = model(data)
    print(output.shape)

    # OUTPUT
    # torch.Size([2, 25088])
    
