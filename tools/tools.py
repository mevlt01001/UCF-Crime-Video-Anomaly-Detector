"""
This file contains tools that AI agent going to use.
"""

# ABNORMAL EVENT DETECTOR PARAMETERS
MODEL_NAME          = "s3d" # Model to video classification
AUGMMENTATION       = None  # No augmentation on inference
CLIP_SIZE           = 16    # Number of frame per clip
OVERLAP             = 8     # Number of overlapped frames between clips
BATCH_SIZE          = 8     # Number of inferenced segments at a time
MAX_NUMBER_OF_CLIPS = 36    # Maxmimum number of clips per segment
MIN_NUMBER_OF_CLIPS = 4     # Minimum number of clips per segments
MICRO_CLIP_BATCH    = 4     # Number of clips per batch in inference tensor
FC_LAYER_CHECKPOINT = None  # Fully connected layer checkpoint
FPS                 = 30    # Frame per seconds for vidoes
MAX_VIDEO_MINUTE    = 10    # Maximum video lenght to process in minute
NUMBER_OF_SEGMENTS  = 32    # Optiimal numer of segments
VIDEO_WIDTH         = 224   # Width for videos
VIDEO_HEIGHT        = 224   # Height for videos
# SAVE_DIR = "anormal_videos_s3d_feats"

import torch
from utils import Video_Analyzer

def abnormal_event_detection(video_path:os.PathLike):
    """
    Bu araç belirli bir video aralığındaki anormal eventların tespit edebilmek amacı ile geliştirilmiştir.
    """

    analyzer = Video_Analyzer(
        MODEL_NAME, CLIP_SIZE, 
        OVERLAP, NUMBER_OF_SEGMENTS, 
        MAX_NUMBER_OF_CLIPS, MIN_NUMBER_OF_CLIPS, 
        FC_LAYER_CHECKPOINT, enable_feature_extractor=True, enable_fc_layers=True, 
        micro_batch_size=MICRO_CLIP_BATCH
    )

    abnormal_segments = analyzer.analyze(
        video_path,VIDEO_WIDTH, VIDEO_HEIGHT, FPS, BATCH_SIZE
    )

    pass