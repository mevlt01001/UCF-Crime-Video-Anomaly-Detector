from __future__ import annotations

import gc
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import math
import torch
import threading
import numpy as np
import torchvision.io as io
import torch.nn.functional as F
import torch.multiprocessing as mp

from tqdm import tqdm
from queue import Queue
from decord import VideoReader, cpu
from torchvision.transforms import v2
from typing import Generator, Optional
from torchvision.models.video import *
from torch.utils.data import Dataset, DataLoader
from .fc_model import SegmentRankingModel
from .visualization_tools import plot_anomaly_timeline
from .video_process import get_video_length, save_segment_clips

model_creaters = {
    "mvit_v1_b": mvit_v1_b,
    "mvit_v2_s": mvit_v2_s,
    "r3d_18": r3d_18,
    "mc3_18": mc3_18,
    "r2plus1d_18": r2plus1d_18,
    "s3d": s3d,
    "swin3d_t": swin3d_t,
    "swin3d_s": swin3d_s,
    "swin3d_b": swin3d_b
}

model_weights = {
    "mvit_v1_b": MViT_V1_B_Weights,
    "mvit_v2_s": MViT_V2_S_Weights,
    "r3d_18": R3D_18_Weights,
    "mc3_18": MC3_18_Weights,
    "r2plus1d_18": R2Plus1D_18_Weights,
    "s3d": S3D_Weights,
    "swin3d_t": Swin3D_T_Weights,
    "swin3d_s": Swin3D_S_Weights,
    "swin3d_b": Swin3D_B_Weights
}

MODEL_SPECIFY_ERROR = lambda model_name : ValueError(f"Specified model {model_name} is unavailable. \
Please use fallowing one: {model_creaters.keys()}")
EMPTY_MODEL_EXPORT_ERROR = lambda enable_feature_extractor, enable_fc_layers : AttributeError(f"An \
empty model cannot be exported! (enable_feature_extractor: {enable_feature_extractor}, enable_fc_layers:{enable_fc_layers}))")
ONNX_PARSING_ERROR = lambda filename : RuntimeError(f"ONNX Parsing error for model: {filename}")
TRT_RUNTIME_GENERATOR_FAILURE = RuntimeError("TRT Plan generation failed!")
MODEL_HAS_NO_CLASSIFIER_ERROR = ValueError("The feature extractor model does not have a classifier or head attribute.")
MAX_VIDEO_MINUTE_ERROR = lambda max_video_minute, video_minute: MemoryError(f"Maximum alloved video lenght {max_video_minute} min. Got video lenght {video_minute} min!")
NO_SEGMENT_ERROR = ValueError("There is no segment in this video.")
VIDEO_TOO_SHORT_ERROR = lambda vp, f, s : ValueError(f"Video {vp} is too short ({f} frames) to be divided into {s} segments.")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unwrap_fc_state_dict(checkpoint):
    if isinstance(checkpoint, (str, os.PathLike)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("FC checkpoint dict veya .pt yolu olmalı.")
    return checkpoint


class Video_Analyzer(torch.nn.Module):
    def __init__(self,
                video_classifier_model: Optional[MViT | VideoResNet | S3D | SwinTransformer3d | str],
                clip_size: int, # Number of frame per clip
                overlap: int,
                fc_layer_checkpoint=None,
                ):

        super().__init__()
        self.clip_size = clip_size
        self.overlap = overlap
        self.stride = clip_size - overlap

        self.__set_video_classifier(video_classifier_model)
        self.__set_video_classifier_feature_dim()
        self.segment_ranker_model = SegmentRankingModel(self.feature_dim)
        if fc_layer_checkpoint:
            self.segment_ranker_model.load_state_dict(
                unwrap_fc_state_dict(fc_layer_checkpoint)
            )

    @staticmethod
    @torch.no_grad()
    def clip_generator(video_path: os.PathLike,
                       clip_size: int,
                       stride: int,
                       fps: int = 30,
                       width: int = 224,
                       height: int = 224,
                       max_video_min: int = 45) -> tuple[Generator[torch.Tensor, None, None], int]:

        vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height, num_threads=2)
        frame_indices = torch.arange(0, len(vr), step=vr.get_avg_fps() / fps).long()
        total_frames = len(frame_indices)
        video_min = total_frames/fps/60

        if max_video_min<video_min:
            raise MAX_VIDEO_MINUTE_ERROR(max_video_min, video_min)

        if total_frames < clip_size:
            raise ValueError(f"Video too short. Total frame ({total_frames}) < clip_size ({clip_size})")

        number_of_clips = 1 + (total_frames - clip_size) // stride

        def _generator(video_reader):
            try:
                for end_idx in range(clip_size, total_frames, stride):
                    start_idx = end_idx - clip_size

                    clip_indices = frame_indices[start_idx:end_idx]
                    frames = video_reader.get_batch(clip_indices.tolist()).asnumpy()

                    clip = torch.from_numpy(frames) # [T, H, W, C]
                    yield clip
            finally:
                del video_reader
                gc.collect()

        return _generator(vr), number_of_clips

    def feature_extractor_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.feature_extractor(x) # [B, dim]
        return x

    def fc_layers_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, dim]
        x = self.segment_ranker_model(x) # [B, 1]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: [B, C, T, H, W]
        if hasattr(self, "trt_context"):
            return self.trt_forward(x)
        x = self.feature_extractor_forward(x) # [B, dim]
        x = self.fc_layers_forward(x) # [B, 1]
        return x

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [B, T, H, W, C]
        x = x.float() / 255.0
        x = x.permute(0, 4, 1, 2, 3).contiguous() # [B, C, T, H, W]
        return x

    @torch.no_grad()
    def analyze(self, 
                video_path: str, 
                width: int,
                height: int, 
                fps: float,
                batch_size: int,
                threshold: float = 0.3,
                tolerance_sec: float = 3.0,
                padding_sec: float = 2.0,
                save_graph: bool = True,
                save_clips: bool = True,
                save_dir: str = "Video_Analyses"
                ):

        self.eval()
        is_trt = hasattr(self, "trt_context") and self.trt_context is not None
        if is_trt:
            device = torch.device("cuda")
        else:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = pick_device()

        os.makedirs(save_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        clip_generator, total_clips = self.clip_generator(
            video_path=video_path, 
            clip_size=self.clip_size, 
            stride=self.stride, 
            fps=fps, 
            width=width, 
            height=height
        )

        mini_batch = [] 
        scores = [] 
        pbar = tqdm(total=total_clips, desc=f"Processing {video_name}")
        
        try:
            for clip in clip_generator: # [T, H, W, C]
                mini_batch.append(clip)
                
                if len(mini_batch) == batch_size:
                    batch_tensor = torch.stack(mini_batch).to(device) # [B, T, H, W, C]
                    batch_tensor = self.preprocess(batch_tensor) # [B, C, T, H, W] (T = clip_size)
                    
                    batch_scores = self.forward(batch_tensor).detach().cpu().float()
                    scores.append(batch_scores)
                    
                    pbar.update(len(mini_batch))
                    mini_batch = []
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            if len(mini_batch) > 0:
                batch_tensor = torch.stack(mini_batch).to(device)
                batch_tensor = self.preprocess(batch_tensor)
                
                batch_scores = self.forward(batch_tensor).detach().cpu().float()
                scores.append(batch_scores)
                
                pbar.update(len(mini_batch))

        except Exception as e:
            print(f"\nGot an error [{video_path}]: {e}")
            raise e
        finally:
            pbar.close()

        return self.create_clips(
                    torch.concat(scores, dim=0), 
                    get_video_length(video_path),
                    video_path,
                    save_dir,
                    threshold,
                    tolerance_sec,
                    padding_sec,
                    save_graph,
                    save_clips
                )

    @torch.no_grad()
    def create_clips(self, 
                        scores: torch.Tensor,
                        video_seconds: float,
                        video_path: str,
                        save_dir: str,
                        threshold: float = 0.3, 
                        tolerance_sec: float = 3.0, 
                        padding_sec: float = 3.0,
                        save_graph: bool = False,
                        save_clips: bool = False,
                        ):
        
        scores_org = scores.float().squeeze(-1) # [NxB]
        scores_org = scores_org.unsqueeze(0).unsqueeze(0)  # [1,1,NxB]
        
        kernel_size = 11
        interpolate_size = max(len(scores.squeeze(-1)), int(video_seconds * 10)) 
        
        scores_linear_interpolate  = F.interpolate(scores_org, size=interpolate_size, mode='linear', align_corners=True)
        scores_nearest_interpolate = F.interpolate(scores_org, size=interpolate_size, mode='nearest')
        
        scores_linear_interpolate  = F.pad(scores_linear_interpolate, (kernel_size // 2, kernel_size // 2), mode='reflect')
        scores_linear_interpolate  = F.avg_pool1d(scores_linear_interpolate, kernel_size=kernel_size, stride=1)
        
        scores_linear_interpolate  = scores_linear_interpolate.squeeze(0).squeeze(0)
        scores_nearest_interpolate = scores_nearest_interpolate.squeeze(0).squeeze(0)
        
        dt = video_seconds / interpolate_size
        anomaly_indices = torch.where(scores_linear_interpolate >= threshold)[0].tolist()
        
        final_segments = []
        
        if anomaly_indices:
            raw_segments = []
            current_start = anomaly_indices[0]
            current_end = anomaly_indices[0]

            for idx in anomaly_indices[1:]:
                time_gap = (idx - current_end) * dt

                if time_gap <= tolerance_sec:
                    current_end = idx
                else:
                    raw_segments.append((current_start, current_end))
                    current_start = idx
                    current_end = idx

            raw_segments.append((current_start, current_end))
            
            padded_segments = []
            for start_idx, end_idx in raw_segments:
                start_time = start_idx * dt
                end_time = end_idx * dt
                
                p_start = max(0.0, start_time - padding_sec)
                p_end = min(video_seconds, end_time + padding_sec)
                
                padded_segments.append([p_start, p_end, start_idx, end_idx])
            
            merged_segments = []
            curr_p_start, curr_p_end, curr_s_idx, curr_e_idx = padded_segments[0]

            for p_start, p_end, s_idx, e_idx in padded_segments[1:]:
                if p_start <= curr_p_end:
                    curr_p_end = max(curr_p_end, p_end)
                    curr_e_idx = e_idx
                else:
                    merged_segments.append((curr_p_start, curr_p_end, curr_s_idx, curr_e_idx))
                    curr_p_start, curr_p_end, curr_s_idx, curr_e_idx = p_start, p_end, s_idx, e_idx
            
            merged_segments.append((curr_p_start, curr_p_end, curr_s_idx, curr_e_idx))

            for p_start, p_end, s_idx, e_idx in merged_segments:
                segment_scores = scores_linear_interpolate[s_idx : e_idx + 1]
                max_score = segment_scores.max().item()

                final_segments.append({
                    "start_time": round(p_start, 2),
                    "end_time": round(p_end, 2),
                    "duration": round(p_end - p_start, 2),
                    "score": round(max_score, 4)
                })

        save_paths = []

        if save_clips or save_graph:
            basename = os.path.basename(video_path)
            save_root = os.path.join(save_dir, os.path.splitext(basename)[0])
            os.makedirs(save_root, exist_ok=True)

            if save_clips:
                save_paths = save_segment_clips(video_path,
                                                final_segments,
                                                save_root)
            if save_graph:
                plot_anomaly_timeline(scores_linear_interpolate.cpu(),
                                      scores_nearest_interpolate.cpu(),
                                      final_segments,
                                      video_seconds,
                                      threshold,
                                      os.path.join(save_root, "segmentation_graph.png"))

        if save_paths:
            for segment, clip_path in zip(final_segments, save_paths):
                segment["clip"] = clip_path

        return final_segments

    @torch.no_grad()
    def trt_forward(self, x: torch.Tensor) -> torch.Tensor:

        if not hasattr(self, "trt_context") or self.trt_context is None:
            raise RuntimeError("TensorRT context not found!")

        if x.device.type != "cuda":
            x = x.to("cuda", non_blocking=True)
            
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x.contiguous()

        B, C, T, H, W = x.shape
        stream = torch.cuda.current_stream()

        if hasattr(self.trt_context, "set_tensor_address"):
            self.trt_context.set_input_shape("video_segment", (B, C, T, H, W))
            output = torch.empty((B, 1), dtype=torch.float32, device="cuda")
            
            self.trt_context.set_tensor_address("video_segment", x.data_ptr())
            self.trt_context.set_tensor_address("score", output.data_ptr())
            self.trt_context.execute_async_v3(stream_handle=stream.cuda_stream)

        return output
    
    def export_onnx(self, file_path: str, batch_size: int, imgsz: int | tuple[int, int] = 224) -> tuple:
        import onnx
        import onnxsim

        self.eval().cpu()
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        B = batch_size
        T = self.clip_size
        H, W = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        
        shape = [B, 3, T, H, W]
        
        dummy_input = torch.rand(*shape, dtype=torch.float32, device="cpu")

        torch.onnx.export(
            self,
            dummy_input,
            file_path,
            input_names=["video_segment"],
            output_names=["score"],
            dynamic_axes={
                'video_segment': {0: 'batch_size'}, 
                'score': {0: 'batch_size'}
            },
            opset_version=19,
            do_constant_folding=True,
        )

        onnx_model = onnx.load(file_path)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        
        onnx_model, check = onnxsim.simplify(onnx_model, test_input_shapes={"video_segment": [B, 3, T, H, W]})
        if check:
            onnx.save(onnx_model, file_path)
        else:
            print("[Warn] ONNXSIM error")

        return shape

    def export_trt(self, 
                   model_path:os.PathLike,
                   batch_size: int, 
                   imgsz: int | tuple[int, int] = 224):
        if not torch.cuda.is_available():
            print("[info] TensorRT NVIDIA CUDA ister; bu cihazda atlandı (MPS/CPU).")
            return None
        import tensorrt as trt

        self.eval().cpu()
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

        dirname = os.path.dirname(model_path)
        os.makedirs(dirname, exist_ok=True)
        trt_file_path = model_path + ".plan"
        onnx_file_path = model_path + ".onnx"
        

        if os.path.exists(trt_file_path):
            print(f"TRT Plan file found, loading: {trt_file_path}")
            return self.load_trt(trt_file_path, trt.Runtime(TRT_LOGGER))
            
        shape = self.export_onnx(onnx_file_path, batch_size, imgsz)
        _, C, T, H, W = shape

        min_shape = (1, C, T, H, W)
        opt_shape = (max(1, batch_size // 2), C, T, H, W)
        max_shape = (batch_size, C, T, H, W)

        def build_engine(onnx_path, plan_path):
            builder = trt.Builder(TRT_LOGGER)

            # explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network()
            parser = trt.OnnxParser(network, TRT_LOGGER)
            config = builder.create_builder_config()
            
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError(f"ONNX Parsing Error in: {onnx_path}")
            
            profile = builder.create_optimization_profile()
            profile.set_shape("video_segment", min=min_shape, opt=opt_shape, max=max_shape)
            config.add_optimization_profile(profile)

            # if fp16 and builder.platform_has_fast_fp16:
            #     config.set_flag(trt.BuilderFlag.FP16)
            
            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                raise RuntimeError(f"TRT Plan generation failed for: {onnx_path}")

            with open(plan_path, "wb") as f:
                f.write(engine_bytes)
            print(f"TRT Plan created: {plan_path}")

        build_engine(onnx_file_path, trt_file_path)
        self.load_trt(trt_file_path, trt.Runtime(TRT_LOGGER))

    def load_trt(self, file_name: os.PathLike, runtime=None):
        import tensorrt as trt
        
        if runtime is None:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            
        with open(file_name, 'rb') as f:
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            
        self.trt_context = self.trt_engine.create_execution_context()
        torch.cuda.empty_cache()

    def __set_video_classifier(self, video_classifier:str|MViT|VideoResNet|S3D|SwinTransformer3d):
        """
        Sets `feature_extractor` model. 
        """
        if isinstance(video_classifier, str): 
            # If specified model by name in formed str
            if video_classifier.lower() in model_creaters.keys():
            # If specified model name available
                self.video_classifier_name = video_classifier.lower()
                self.feature_extractor = model_creaters[video_classifier](weights=model_weights[video_classifier].DEFAULT)
            else: 
                # If specified model name does not available
                raise MODEL_SPECIFY_ERROR(video_classifier)
        elif isinstance(video_classifier, (MViT, VideoResNet, S3D, SwinTransformer3d)):
            # If Specified model formed its class form
            self.feature_extractor = video_classifier
            self.video_classifier_name = video_classifier.__class__.__name__.lower()
        else:
            # If specified model does not met any condition:
             raise ValueError(MODEL_SPECIFY_ERROR(video_classifier))
        
    def __set_video_classifier_feature_dim(self):
        """
        Sets `feature_dim`.
        """
        if hasattr(self.feature_extractor, "head"): # MViT and SwinTransformer3d
            if isinstance(self.feature_extractor.head, torch.nn.Linear):
                self.feature_dim = self.feature_extractor.head.in_features
                self.feature_extractor.head = torch.nn.Identity()
            elif isinstance(self.feature_extractor.head, torch.nn.Sequential):
                self.feature_dim = self.feature_extractor.head[1].in_features
                self.feature_extractor.head = torch.nn.Identity()

        elif hasattr(self.feature_extractor, "fc"): # VideoResNet
            self.feature_dim = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = torch.nn.Identity()

        elif hasattr(self.feature_extractor, "classifier"): # S3D
            self.feature_dim = self.feature_extractor.classifier[1].in_channels
            self.feature_extractor.classifier = torch.nn.Identity()
        else:
            raise MODEL_HAS_NO_CLASSIFIER_ERROR

@torch.inference_mode()
def extract_feats(analyzer: Video_Analyzer,
                  video_paths: list,
                  save_dir: str,
                  fps: int,
                  width: int,
                  height: int,
                  batch: int,
                  max_video_min: int,
                  save_debug_video: bool,
                  augmentation: v2.Compose = None):
    
    extractor = analyzer.feature_extractor
    H, W = height, width

    skip_log_path = os.path.join(save_dir, "skipped_videos.txt")
    os.makedirs(save_dir, exist_ok=True)
 
    num_gpus = torch.cuda.device_count()
    is_dp = num_gpus > 1
    
    if is_dp: 
        extractor = torch.nn.DataParallel(extractor)
 
    extractor = extractor.to("cuda")
    extractor.eval()
 
    len_videos = len(video_paths)
 
    def print_info(video_idx, vp, current_clip, total_clips):
        info = (
            f"Video: {video_idx + 1:04d}/{len_videos} [{os.path.basename(vp)}] | "
            f"Processed Clips: {current_clip:03d}/{total_clips} | "
        )
        print(info, end='\n', flush=True)
 
    for video_idx, vp in enumerate(video_paths):
        base_name = os.path.splitext(os.path.basename(vp))[0]
        save_path = os.path.join(save_dir, f"{base_name}.pt")
        
        video_features = [] # [N, Dim]
        debug_writer = None
        debug_save_path = None
 
        try:
            clip_generator, total_clips = Video_Analyzer.clip_generator(
                video_path=vp, 
                clip_size=analyzer.clip_size,
                stride=analyzer.stride,
                fps=fps, 
                width=W, 
                height=H,
                max_video_min=max_video_min
            )

            mini_batch = []
            
            if save_debug_video:
                debug_save_path = os.path.join(save_dir, f"{base_name}_debug.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                debug_writer = cv2.VideoWriter(debug_save_path, fourcc, fps, (W, H))

            for clip_idx, clip in enumerate(clip_generator):

                if augmentation:
                    clip = clip.permute(0, 3, 1, 2)
                    clip = augmentation(clip)
                    clip = clip.permute(0, 2, 3, 1)
                    
                if save_debug_video and debug_writer is not None:
                    T_frames, h_clip, w_clip, C_channels = clip.shape
                    clip_np = clip.cpu().numpy()
                    
                    for frame_idx in range(T_frames):
                        frame_data = clip_np[frame_idx]
                        if frame_data.dtype != np.uint8:
                            if frame_data.max() <= 1.0: 
                                frame_data = (frame_data * 255.0)
                            frame_data = np.clip(frame_data, 0, 255).astype(np.uint8)

                        if C_channels == 3:
                            frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
                        else:
                            frame_bgr = frame_data
                        
                        text = f"Clip:{clip_idx+1}/{total_clips}\nFrame:{frame_idx+1}/{T_frames}"
                        y0, dy = 20, 20
                        for i, line in enumerate(text.split('\n')):
                            y = y0 + i * dy
                            cv2.putText(frame_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                        debug_writer.write(frame_bgr)


                mini_batch.append(clip)
                if len(mini_batch) == batch:
                    batch_tensor = torch.stack(mini_batch).to("cuda") # [B, T, H, W, C]
                    batch_tensor = analyzer.preprocess(batch_tensor)  # [B, C, T, H, W]
                    feats = extractor(batch_tensor) 
                    video_features.append(feats.detach().cpu())
                    
                    print_info(video_idx, vp, clip_idx + 1, total_clips)
                    
                    del batch_tensor
                    mini_batch.clear()
                    torch.cuda.empty_cache()

            if len(mini_batch) > 0:
                batch_tensor = torch.stack(mini_batch).to("cuda")
                batch_tensor = analyzer.preprocess(batch_tensor)
                
                if hasattr(analyzer, 'transforms') and analyzer.transforms is not None:
                    batch_tensor = analyzer.transforms(batch_tensor)
                
                feats = extractor(batch_tensor) 
                video_features.append(feats.detach().cpu())
                print_info(video_idx, vp, total_clips, total_clips)
                
            print()
 
            if len(video_features) > 0:
                video_feature_tensor = torch.cat(video_features, dim=0) 
                
                data = {
                    "feats": video_feature_tensor,
                    "model": getattr(analyzer, 'video_classifier_name', "Unknown"),
                    "clip_size": analyzer.clip_size,
                    "overlap": analyzer.overlap,
                    "fps": fps,
                    "width": W,
                    "height": H,
                    "augmentation": str(augmentation) if augmentation is not None else None
                }
                torch.save(data, save_path)
            else:
                raise ValueError(f"No valid clips extracted from video: {vp}")
            
            if save_debug_video and debug_writer is not None:
                debug_writer.release()
                debug_writer = None 
                print(f"[DEBUG] Video saved at: {debug_save_path}")
 
            del video_feature_tensor, video_features, clip_generator, data
            gc.collect()
            torch.cuda.empty_cache()
 
        except Exception as e:
            if save_debug_video and debug_writer is not None:
                debug_writer.release()
                debug_writer = None
                
            with open(skip_log_path, "a", encoding="utf-8") as f:
                f.write(f"{vp}\t{str(e)}\n")
            print(f"\n[Skipped] {os.path.basename(vp)} -> Err: {str(e)}", flush=True)
            
            gc.collect()
            torch.cuda.empty_cache()