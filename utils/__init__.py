from .llm import LLM_Manager

__all__ = [
    "LLM_Manager",
    "VLM_Manager",
    "SegmentRankingModel",
    "segment_score_model_trainer",
    "Video_Analyzer",
    "extract_feats",
    "fetch_video_patches",
    "get_video_length",
    "save_segment_clips",
    "get_report_dir",
    "generate_frames",
]
from .video_process import (
    fetch_video_patches, 
    get_video_length, 
    save_segment_clips, 
    get_report_dir, 
    generate_frames
)

_LAZY = {
    "VLM_Manager": (".vlm", "VLM_Manager"),
    "SegmentRankingModel": (".fc_model", "SegmentRankingModel"),
    "segment_score_model_trainer": (".fc_model", "segment_score_model_trainer"),
    "Video_Analyzer": (".video_analyzer_model", "Video_Analyzer"),
    "extract_feats": (".video_analyzer_model", "extract_feats"),
    "fetch_video_patches": (".video_process", "fetch_video_patches"),
    "get_video_length": (".video_process", "get_video_length"),
    "save_segment_clips": (".video_process", "save_segment_clips"),
    "get_report_dir": (".video_process", "get_report_dir"),
    "generate_frames": (".video_process", "generate_frames"),
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    from importlib import import_module
    return getattr(import_module(module_name, __package__), attr)
