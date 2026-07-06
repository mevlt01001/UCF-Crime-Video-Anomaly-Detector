import torch
import torch.nn as nn
import torch.nn.functional as F
from .c3d import C3D

class C3D_FeatureExtractor(nn.Module):

    def __init__(self, pt_file:str, clip_size: int = 16, overlap: int = 8):
        super(C3D_FeatureExtractor, self).__init__()
        self.clip_size = clip_size
        self.overlap = overlap
        self.stride = clip_size - overlap

        self.C3D = C3D()
        self.C3D.load_state_dict(torch.load(pt_file))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape: (B, C, S, H, W) -> Tamamen dinamik SEQ desteği
        """
        B, C, S, H, W = x.shape

        pad_size = max(0, self.clip_size - S)
        x = F.pad(x, (0, 0, 0, 0, 0, pad_size))
        S = x.shape[2]

        x_unfolded = x.unfold(2, self.clip_size, self.stride)
        
        x_clips = x_unfolded.permute(0, 2, 1, 5, 3, 4)
        num_clips = x_clips.shape[1]

        h = x_clips.reshape(-1, C, self.clip_size, H, W)

        h = self.C3D.extract(h, layer="conv5")

        h = h.reshape(B, num_clips, -1)
        h = h.mean(dim=1)

        return h

if __name__ == "__main__":

    data = torch.randn(2, 3, 16, 112, 112).to("cuda")
    model = C3D_FeatureExtractor("C3D.pt").to("cuda")
    output = model(data)
    print(output.shape)

    # OUTPUT
    # torch.Size([2, 25088])