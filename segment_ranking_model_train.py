import torch
from utils import SegmentRankingModel, segment_score_model_trainer

model = SegmentRankingModel(input_dim=512).to("cuda")

segment_score_model_trainer(model=model,
                    anormal_feat_dir="extracted_anormal_features",
                    normal_feat_dir="extracted_normal_features",
                    epochs=400,
                    batch_size=64,
                    learning_rate=0.001,
                    test_ratio=0.1)