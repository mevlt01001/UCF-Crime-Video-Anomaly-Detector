import torch
from utils import SegmentRankingModel, MIL_network_trainer

model = SegmentRankingModel(input_dim=512).to("cuda")

MIL_network_trainer(model=model,
                    anormal_feat_dir="extracted_anormal_features",
                    normal_feat_dir="extracted_normal_features",
                    epochs=500,
                    batch_size=128,
                    learning_rate=0.001,
                    test_ratio=0.1)