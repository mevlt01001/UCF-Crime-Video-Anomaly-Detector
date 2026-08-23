import torch
from utils import SegmentRankingModel, segment_score_model_trainer

model = SegmentRankingModel(input_dim=1024).to("cuda")

segment_score_model_trainer(model=model,
                    anormal_feat_dir="S3D_Feats/no_aug_plus_aug copy/anormal_videos_s3d_feats_no_aug",
                    normal_feat_dir="S3D_Feats/no_aug_plus_aug copy/normal_videos_s3d_feats_no_aug",
                    epochs=100,
                    batch_size=64,
                    learning_rate=0.001,
                    early_stop_patience=12,
                    k_fold=1,
                    weight_decay=0.001,
                    Sparsity_loss_k=0.0001,
                    Smoothness_loss_k=0.0001,
                    Normal_loss_k=0.0001,
                    val_ratio=0.25)