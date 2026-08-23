import torch
from utils import SegmentRankingModel, segment_score_model_trainer

model = SegmentRankingModel(input_dim=1024).to("cuda")

segment_score_model_trainer(model=model,
                    anormal_feat_dir="S3D_Feats/8/anormal_videos_feats",
                    normal_feat_dir="S3D_Feats/8/normal_videos_feats",
                    epochs=75,
                    batch_size=96,
                    learning_rate=0.001,
                    early_stop_patience=8,
                    k_fold=5,
                    weight_decay=0.0025,
                    Sparsity_loss_k=0.0001,
                    Smoothness_loss_k=0.0001,
                    Normal_loss_k=0.0001,
                    val_ratio=0.25)