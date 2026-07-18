import torch
from utils import SegmentRankingModel, segment_score_model_trainer

model = SegmentRankingModel(input_dim=512).to("cuda")

segment_score_model_trainer(model=model,
                    anormal_feat_dir="feats_112x112/extracted_anormal_features",
                    normal_feat_dir="feats_112x112/extracted_normal_features",
                    epochs=100,
                    batch_size=512,
                    learning_rate=0.001,
                    test_ratio=0.25)