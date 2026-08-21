import gc
import os
import cv2
import torch
import threading
import numpy as np
import torchvision.io as io
import torch.nn.functional as F

from tqdm import tqdm
from queue import Queue
from decord import VideoReader, cpu
from torchvision.transforms import v2
from typing import Generator, Optional
from torchvision.models.video import *
from .SegmentRankingModel import SegmentRankingModel
from .visualization_tools import plot_anomaly_timeline
from .video_preprocess import get_video_length, save_segment_clips, AddGaussianNoise, AddSaltAndPepperNoise

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

def get_number_of_segments(min_segment_size: int,
                            max_segment_size: int,
                            number_of_segment: int,
                            total_frames: int,
                            clip_size: int,
                            stride: int = None) -> int:
    """
    Calculates the optimum number of segments based on the specified clip limit.
    """
    if stride is None:
        stride = clip_size
 
    max_frame_per_segment = ((max_segment_size - 1) * stride) + clip_size
    min_frame_per_segment = ((min_segment_size - 1) * stride) + clip_size
 
    min_number_of_segment = total_frames // max_frame_per_segment + 1
    max_number_of_segment = total_frames // min_frame_per_segment - 1
 
    max_number_of_segment = max(1, max_number_of_segment)
 
    if number_of_segment <= min_number_of_segment:
        number_of_segment = min_number_of_segment
 
    if number_of_segment > max_number_of_segment:
        number_of_segment = max_number_of_segment
 
    return max(1, number_of_segment)


class Video_Feature_Extractor(torch.nn.Module):
    """
    This module is designed to extract frames by pre-trained video classifier models.
    
    :param feature_extractor_model: Pre-trained video classification model.
    :param int segments_per_video: Number of segments to strictly divide the video into.
    """
    def __init__(self,
                feature_extractor_model: Optional[MViT | VideoResNet | S3D | SwinTransformer3d | str],
                clip_size: int,
                overlap: int,
                number_of_segments: int,
                max_clip_per_segment: int,
                min_clip_per_segment: int,
                fc_layer_checkpoint=None,
                enable_feature_extractor: bool = True,
                enable_fc_layers: bool = True,
                micro_batch_size: int = 8,
                ):
        
        super().__init__()
        self.clip_size = clip_size
        self.overlap = overlap
        self.stride = clip_size - overlap
        self.number_of_segments = number_of_segments
        self.max_clip_per_segment = max_clip_per_segment
        self.min_clip_per_segment = min_clip_per_segment
        self.enable_feature_extractor = enable_feature_extractor
        self.enable_fc_layers = enable_fc_layers

        self.micro_batch_size = micro_batch_size

        self.__set_video_classifier(feature_extractor_model)
        self.__set_video_classifier_feature_dim()
        self.segment_ranker_model = SegmentRankingModel(self.feature_dim)
        if fc_layer_checkpoint:
            self.segment_ranker_model.load_state_dict(fc_layer_checkpoint)
        self.std = model_weights[self.video_classifier_name].DEFAULT.transforms().std
        self.mean = model_weights[self.video_classifier_name].DEFAULT.transforms().mean

    @staticmethod
    def clip_generator(segment_frames: torch.Tensor,
                        clip_size: int,
                        stride: int) -> torch.Tensor:
        """
        :param segment_frames torch.Tensor: [N, H, W, C]
        :param clip_size int: Number of frame per clip
        :param stride int: Slider window step size
        :return torch.Tensor: [NC, CS, H, W, C]
        """
        N = segment_frames.shape[0]
 
        if N < clip_size:
            pad = clip_size - N
            segment_frames = F.pad(segment_frames, (0, 0, 0, 0, 0, 0, 0, pad))
            N = segment_frames.shape[0]
 
        remain = (N - clip_size) % stride
        if remain != 0:
            pad = stride - remain
            if pad >= remain:
                segment_frames = segment_frames[: N - remain]
            else:
                segment_frames = F.pad(segment_frames, (0, 0, 0, 0, 0, 0, 0, pad))
            N = segment_frames.shape[0]
 
        # [N, H, W, C] -> unfold(dim=0) -> [NC, H, W, C, CS] -> permute -> [NC, CS, H, W, C]
        clips = segment_frames.unfold(0, clip_size, stride)
        clips = clips.permute(0, 4, 1, 2, 3).contiguous()
        return clips

        

    @staticmethod
    def segment_generator(video_path: os.PathLike,
                           clip_size: int,
                           stride: int,
                           max_clips_per_segment: int | None,
                           min_clips_per_segment: int | None,
                           number_of_segments: int,
                           fps: int = 30,
                           max_video_minute:int = 20,
                           width: int = 224,
                           height: int = 224) -> tuple[Generator[torch.Tensor, None, None], int]:
        """
        """
        vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height, num_threads=4)
        org_fps = vr.get_avg_fps()
        frame_indices = torch.arange(0, len(vr), step=org_fps / fps).long()
        total_frames = len(frame_indices)
        video_minute = (total_frames/fps)/60
        total_segments = get_number_of_segments(min_clips_per_segment, max_clips_per_segment,
                                                number_of_segments, total_frames, clip_size, stride)

        if max_video_minute < video_minute:
            raise MAX_VIDEO_MINUTE_ERROR(max_video_minute, video_minute)
 
        if total_frames < total_segments * clip_size:
            raise VIDEO_TOO_SHORT_ERROR(video_path, total_frames, total_segments)
 
        segment_size = total_frames // total_segments
 
        def _generator(video_reader):
            try:
                for i in range(total_segments):
                    start_idx = i * segment_size
                    end_idx = start_idx + segment_size
     
                    segment_indices = frame_indices[start_idx:end_idx]        
                    segment_frames = video_reader.get_batch(segment_indices).asnumpy()  
                    segment_tensor = torch.from_numpy(segment_frames)         
     
                    segment_tensor = Video_Feature_Extractor.clip_generator(  
                        segment_tensor, clip_size, stride
                    )
                    segment_tensor = segment_tensor.permute(0, 4, 1, 2, 3).contiguous()  
                    
                    yield segment_tensor
            finally:
                del video_reader
 
        return _generator(vr), total_segments


    def feature_extractor_forward(self, x: torch.Tensor, augmentation: v2.Compose = None, save_video: bool = False) -> torch.Tensor:
        B, NC, C, CS, H, W = x.shape
        device = x.device if x.is_cuda else next(self.parameters()).device
        chunk_size = max(1, self.micro_batch_size)
        feat_sum = None
        x = x.to(device, non_blocking=True).float().div_(255.0)

        if augmentation:
            augmented_x = []
            for b in range(B):
                segment = x[b].permute(0, 2, 1, 3, 4).reshape(NC*CS, C, H, W) # [NC, C, CS, H, W] -> [NC, CS, C, H, W] -> [N, C, H, W]
                segment = augmentation(segment)
                segment = segment.reshape(NC, CS, C, H, W).permute(0, 2, 1, 3, 4) # [N, C, H, W] -> [NC, CS, C, H, W] -> [NC, C, CS, H, W]
                augmented_x.append(segment)
            x = torch.stack(augmented_x, dim=0) # [B, NC, CS, C, H, W]

        debug_clips = [] # n X [B, chunk_size, C, CS, H, W]

        for start in range(0, NC, chunk_size):
            end = min(start + chunk_size, NC)
            chunk = x[:, start:end]  # [B, NCC, C, CS, H, W] uint8
            NCC = chunk.shape[1] # Number of Clips per Chunk
            
            chunk = chunk.reshape(B * NCC, C, CS, H, W)

            if save_video:
                chunk_debug = chunk.reshape(B, NCC, C, CS, H, W)
                debug_clips.append(chunk_debug.clone().cpu())

            mean = torch.tensor(self.mean, device=device).view(1, 3, 1, 1, 1)
            std = torch.tensor(self.std, device=device).view(1, 3, 1, 1, 1)
            chunk = (chunk - mean) / std

            # [B*NCC, Dim, ...]
            if hasattr(self, "trt_engine_feat") and self.trt_engine_feat is not None:
                feats = self.trt_inference_feat(chunk) 
            else:
                feats = self.feature_extractor(chunk)

            feats = feats.view(B, NCC, -1).float().sum(dim=1)
            feat_sum = feats if feat_sum is None else feat_sum + feats
            
            del chunk, feats

        x = feat_sum / NC
        if debug_clips:
            debug_clips = torch.concat(debug_clips, dim=1) # [B, NC, C, CS, H, W]

        return x, debug_clips

    def fc_layers_forward(self, x):
        # x.shape is [B, dim]
        # x.shape is [B, dim]
        if hasattr(self, "trt_engine_fc") and self.trt_engine_fc is not None:
            x = self.trt_inference_fc(x)
        else:
            x = self.segment_ranker_model(x)  # [B, 1]
            
        return x
 
    def forward(self, x: torch.Tensor, augmentation: v2.Compose = None, save_debug_clip:bool=False) -> torch.Tensor:
        """
        :param x  torch.Tensor: Input data to process by model. **Shape information:** \
            if `enable_feature_extractor` its shape must be `[B, NC, C, CS, H, W]`(`B`: Number of Segemnts, `NC`: Number of Clips, `CS`: Clip Size) \
                if not `enable_feature_extractor` and `enable_fc_layers` its shape must be `[B, Dim]`(`B`: Number of Segemnts, `Dim`: Feature Dimension Size)
        :param augmentation torchvision.transforms.v2.Compose: Augmentation compose from torchvision.transforms.v2.Compose. Used when feature extracting.
        :param save_debug_clip bool: Flag indicating whether to visualize input data passed to the 'feature_extractor'.
        :return tuple(torch.Tensor, list): Returns output data and debug frames. Debug frames will be empty If not `save_debug_clip`.\
            If `enable_fc_layers` the output shape will be [B, 1] and stands for anomaly scores per segment. if not `enable_fc_layers` and `enable_feature_extractor` \
            the output shape will be [B, Dim] and it stands for temporal feature per segment.
        """
        debug_frames = []
        if self.enable_feature_extractor:
            # x.shape = [B, NC, C, CS, H, W] -> [B, Dim]
            x, debug_frames = self.feature_extractor_forward(x, augmentation=augmentation, save_video=save_debug_clip)
        if self.enable_fc_layers:
            # x.shape = [B, Dim] -> [B, 1]
            x = self.fc_layers_forward(x)
 
        return x, debug_frames


    @torch.no_grad()
    def analyze(self, 
                video_path:str, 
                width:int,
                height:int, 
                FPS:float,
                batch_size:int=8,
                threshold=0.3,
                tolerance_sec=3.0,
                padding_sec=2.0,
                save_graph=True,
                save_clips=True,
                save_dir="Video_Analyses"
                ):
        
        from tqdm import tqdm
        from decord import VideoReader, cpu

        def batch_flush(mini_batches):
            # N x [C, T, H, W] -> [N, C, T, H, W]
            mini_batches = torch.stack(mini_batches, dim=0)
            mini_batches = mini_batches.to(device=device, 
                                        dtype=torch.float32, 
                                        non_blocking=True)
            # 0-1 Normalization
            mini_batches = mini_batches.div_(255.0)    
            return mini_batches

        def inference(mini_batches):
            mini_batches = batch_flush(mini_batches)
            
            if self.trt_engine is not None:
                score, _ = self.forward(mini_batches) 
            else:
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    score, _ = self.forward(mini_batches) 
                    
            return score

        if hasattr(self, "trt_engine_fc"): 
            device = "cuda"
        else:
            device = next(self.parameters()).device.type
        if hasattr(self, "trt_engine") and self.trt_engine is not None:
            device = "cuda"

        segment_generator, number_of_segments = Video_Feature_Extractor.segment_generator(
            video_path=video_path, 
            clip_size=self.clip_size, 
            stride=self.stride, 
            max_clips_per_segment=self.max_clip_per_segment, 
            min_clips_per_segment=self.min_clip_per_segment,
            number_of_segments=self.number_of_segments, 
            fps=FPS, 
            width=width, 
            height=height
        )

        total_segments = number_of_segments
        scores = [] # N x [B, 1]
        mini_batches = [] # N x [NC, C, CS, H, W]
        
        pbar = tqdm(total=total_segments, desc=f"Processing {os.path.basename(video_path)}")
        
        try:
            for segment in segment_generator:
                print(segment.shape)
                mini_batches.append(segment)
                
                if len(mini_batches) == batch_size:
                    print("processin1")
                    score = inference(mini_batches) # [B,1]
                    print(score)
                    scores.append(score.detach().float().cpu())
                    
                    pbar.update(batch_size)
                    
                    del score
                    mini_batches.clear()
                    torch.cuda.empty_cache()

            if mini_batches:
                print("processin last")
                score = inference(mini_batches)
                scores.append(score.detach().float().cpu())
                
                pbar.update(len(mini_batches))
                
                del score
                mini_batches.clear()
                torch.cuda.empty_cache()    
                
        except Exception as e:
            print(f"\nGot an error [{video_path}]: {e}")
        finally:
            pbar.close()

        # Create Abnormal Segments
        return self.create_segments(
                    torch.concat(scores, dim=0), # [NxB, 1]
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
    def create_segments(self, 
                        scores:torch.Tensor,
                        video_seconds:float,
                        video_path:os.PathLike,
                        save_dir:os.PathLike,
                        threshold:float=0.3, 
                        tolerance_sec:float=3.0, 
                        padding_sec:float=3.0,
                        save_graph:bool=False,
                        save_clips:bool=False,
                        ):
        
        scores_org = scores.squeeze(-1) # [NxB]
        scores_org = scores_org.unsqueeze(0).unsqueeze(0)  # [1,1,NxB]
        
        kernel_size = 11
        interpolate_size = max(1000, scores_org.shape[-1]*10)
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

        save_paths = None

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
        if save_paths is not None:
            for segment, clip_path in zip(final_segments, save_paths):
                segment["clip"] = clip_path

        return final_segments

    def trt_inference_feat(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """
        Executes TensorRT inference for the Feature Extractor.
        """
        x_chunk = x_chunk.contiguous().to(device="cuda", dtype=torch.float32)
        self.trt_context_feat.set_input_shape("x", tuple(x_chunk.shape))
        output_shape = tuple(self.trt_context_feat.get_tensor_shape("y"))
        output = torch.empty(output_shape, dtype=torch.float32, device=x_chunk.device)

        self.trt_context_feat.set_tensor_address("x", int(x_chunk.data_ptr()))
        self.trt_context_feat.set_tensor_address("y", int(output.data_ptr()))
        
        stream = torch.cuda.current_stream().cuda_stream
        self.trt_context_feat.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()
        return output

    def trt_inference_fc(self, features: torch.Tensor) -> torch.Tensor:
        """
        Executes TensorRT inference for the Fully Connected (Segment Ranker) layer.
        """
        features = features.contiguous().to(device="cuda", dtype=torch.float32)
        self.trt_context_fc.set_input_shape("x", tuple(features.shape))
        output_shape = tuple(self.trt_context_fc.get_tensor_shape("y"))
        output = torch.empty(output_shape, dtype=torch.float32, device=features.device)

        self.trt_context_fc.set_tensor_address("x", int(features.data_ptr()))
        self.trt_context_fc.set_tensor_address("y", int(output.data_ptr()))
        
        stream = torch.cuda.current_stream().cuda_stream
        self.trt_context_fc.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()
        return output

    def export_onnx(self, filename: str, imgsz: tuple|int, batch_size: int):
        """
        Exports both Feature Extractor and FC Layers (Segment Ranker) to separate ONNX files.
        
        :param filename os.PathLike: Base filename to save onnx files. Will automatically append `_feat.onnx` and `_fc.onnx`.
        :param imgsz tuple|int: Video frame size. **If tuple** `H, W = imgsz` **if int** `H, W = imgsz, imgsz`
        :param batch_size int: Number of Segments to inference at a time (Used for FC max batch size estimation).
        :return tuple: `tuple(TRT_MAX_BATCH, BATCH_SIZE, CS, H, W, DIM)` to pass boundary data to TRT Builder.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        import onnx, onnxsim

        CS = self.clip_size
        TRT_MAX_BATCH = self.micro_batch_size * batch_size
        H, W = imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)
        DIM = self.feature_dim

        feat_onnx_path = filename + "_feat.onnx"
        fc_onnx_path = filename + "_fc.onnx"

        # Export Feature Extractor
        if self.enable_feature_extractor:
            dummy_feat = torch.rand(TRT_MAX_BATCH, 3, CS, H, W, device="cpu")
            if getattr(self, "trt_engine_feat", None):
                del self.trt_engine_feat
                
            torch.onnx.export(
                self.feature_extractor.to("cpu"),
                dummy_feat,
                feat_onnx_path,
                input_names=["x"],
                output_names=["y"],
                dynamic_axes={'x': {0: 'Clip_Batch'}},
                opset_version=19,
                do_constant_folding=True,
            )

            onnx_model = onnx.load(feat_onnx_path)
            onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
            onnx_model, check = onnxsim.simplify(onnx_model, test_input_shapes={"x": [TRT_MAX_BATCH, 3, CS, H, W]})
            onnx.save(onnx_model, feat_onnx_path)

        # 2. Export FC Layers
        if self.enable_fc_layers:
            dummy_fc = torch.rand(batch_size, DIM, device="cpu")
            if getattr(self, "trt_engine_fc", None):
                del self.trt_engine_fc

            torch.onnx.export(
                self.segment_ranker_model.to("cpu"),
                dummy_fc,
                fc_onnx_path,
                input_names=["x"],
                output_names=["y"],
                dynamic_axes={'x': {0: 'Segment_Batch'}},
                opset_version=19,
                do_constant_folding=True,
            )

            onnx_model_fc = onnx.load(fc_onnx_path)
            onnx_model_fc = onnx.shape_inference.infer_shapes(onnx_model_fc)
            onnx_model_fc, check = onnxsim.simplify(onnx_model_fc, test_input_shapes={"x": [batch_size, DIM]})
            onnx.save(onnx_model_fc, fc_onnx_path)

        return TRT_MAX_BATCH, batch_size, CS, H, W, DIM

    def export_trt(self, file_name: os.PathLike, imgsz: int|tuple[int, int], batch_size: int):
        """
        Builds separate TensorRT plan files for both Feature Extractor and FC layers.
        
        :param file_name os.PathLike: Base filename to save TensorRT plan files.
        :param imgsz tuple|int: Video frame size. **If tuple** `H, W = imgsz` **if int** `H, W = imgsz, imgsz`
        :param batch_size int: Number of Segments to inference at a time.
        """
        self.eval().cpu()
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        
        feat_plan_path = file_name + "_feat.plan"
        fc_plan_path = file_name + "_fc.plan"

        if os.path.exists(feat_plan_path) and os.path.exists(fc_plan_path):
            return self.load_trt(file_name, trt.Runtime(TRT_LOGGER))
            
        TRT_MAX_BATCH, BATCH_SIZE, CS, H, W, DIM = self.export_onnx(file_name, imgsz, batch_size)

        def build_engine(onnx_path, plan_path, min_shape, opt_shape, max_shape):
            BUILDER = trt.Builder(TRT_LOGGER)
            NETWORK = BUILDER.create_network()
            PARSER = trt.OnnxParser(NETWORK, TRT_LOGGER)
            CONFIG = BUILDER.create_builder_config()
            OPT_PROFILE = BUILDER.create_optimization_profile()
            
            if not PARSER.parse_from_file(onnx_path):
                for error in range(PARSER.num_errors):
                    print(PARSER.get_error(error))
                raise Exception(f"ONNX Parsing Error in: {onnx_path}")
            
            OPT_PROFILE.set_shape("x", min_shape, opt_shape, max_shape)
            CONFIG.add_optimization_profile(OPT_PROFILE)
            
            engine_bytes = BUILDER.build_serialized_network(NETWORK, CONFIG)
            if engine_bytes is None:
                raise Exception(f"TRT Plan generation failed for: {onnx_path}")

            with open(plan_path, "wb") as f:
                f.write(engine_bytes)

        if self.enable_feature_extractor:
            build_engine(
                file_name + "_feat.onnx", feat_plan_path,
                min_shape=(1, 3, CS, H, W),
                opt_shape=(max(1, TRT_MAX_BATCH // 2), 3, CS, H, W),
                max_shape=(TRT_MAX_BATCH, 3, CS, H, W)
            )

        if self.enable_fc_layers:
            build_engine(
                file_name + "_fc.onnx", fc_plan_path,
                min_shape=(1, DIM),
                opt_shape=(max(1, BATCH_SIZE // 2), DIM),
                max_shape=(BATCH_SIZE, DIM)
            )

        self.load_trt(file_name, trt.Runtime(TRT_LOGGER))
            
    def load_trt(self, file_name: os.PathLike, runtime):
        """
        Loads TensorRT engine files already generated before and frees up VRAM by deleting PyTorch models. 
        """
        feat_plan_path = file_name + "_feat.plan"
        fc_plan_path = file_name + "_fc.plan"

        if self.enable_feature_extractor and os.path.exists(feat_plan_path):
            with open(feat_plan_path, 'rb') as f:
                self.trt_engine_feat = runtime.deserialize_cuda_engine(f.read())
            self.trt_context_feat = self.trt_engine_feat.create_execution_context()
            
            if hasattr(self, 'feature_extractor'):
                del self.feature_extractor

        if self.enable_fc_layers and os.path.exists(fc_plan_path):
            with open(fc_plan_path, 'rb') as f:
                self.trt_engine_fc = runtime.deserialize_cuda_engine(f.read())
            self.trt_context_fc = self.trt_engine_fc.create_execution_context()
            
            if hasattr(self, 'segment_ranker_model'):
                del self.segment_ranker_model

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
        elif isinstance (video_classifier, (MViT | VideoResNet | S3D | SwinTransformer3d)):
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
def extract_feats(extractor: Video_Feature_Extractor,
                   video_paths: list,
                   save_dir: str,
                   batch_size: int = 4,
                   imgsz: int | tuple = 224,
                   fps: int = 30,
                   max_video_minute:int = 20,
                   augmentation=None,
                   skip_log_path: str = None,
                   save_debug_video:bool=False,
                   log_every: int = 1):
 
    extractor.enable_feature_extractor = True
    extractor.enable_fc_layers = False
    H, W = imgsz if isinstance(imgsz, (tuple, list)) else (imgsz, imgsz)
 
    os.makedirs(save_dir, exist_ok=True)
    if skip_log_path is None:
        skip_log_path = os.path.join(save_dir, "skipped_videos.txt")
 
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
 
    num_gpus = torch.cuda.device_count()
    is_dp = num_gpus > 1
 
    if is_dp:
        extractor = torch.nn.DataParallel(extractor)
 
    extractor = extractor.to("cuda")
    extractor.eval()
 
    core_model = extractor.module if is_dp else extractor
    len_videos = len(video_paths)
    all_debug_frames = []
    def inference(batch: torch.Tensor):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16 if not save_debug_video else torch.float32):
            feats, debug_frames = extractor(batch, augmentation, save_debug_video)  # [B, Dim]
        all_debug_frames.append(debug_frames)
        return feats
 
    def print_info(video_idx, vp, segment_idx, total_segments, segment):
        info = (
            f"Video: {video_idx + 1:04d}/{len_videos} | "
            f"Video: {os.path.basename(vp)} | "
            f"Segment: {segment_idx + 1:03d}/{total_segments} | "
            f"Clip Size: {segment.shape[2]} | "
            f"Number of Clips: {segment.shape[0]} "
        )
        print(info, end='\n', flush=True)
 
    for video_idx, vp in enumerate(video_paths):
        save_path = os.path.join(save_dir, os.path.basename(vp) + ".pt")
 
        if os.path.exists(save_path):
            print(f"Skipping.. (Already exists): {video_idx + 1:04d}/{len_videos} -> {os.path.basename(vp)}", flush=True)
            continue
 
        video_features = []  # N x [B, Dim]
        batch_buffer = []    # B x [NC, C, CS, H, W] uint8
 
        try:
            segment_generator, total_segments = Video_Feature_Extractor.segment_generator(
                video_path=vp,
                clip_size=core_model.clip_size,
                stride=core_model.stride,
                max_clips_per_segment=core_model.max_clip_per_segment,
                min_clips_per_segment=core_model.min_clip_per_segment,
                number_of_segments=core_model.number_of_segments,
                fps=fps,
                max_video_minute=max_video_minute,
                width=W,
                height=H,
            )
 
            segment_idx = -1
            for segment_idx, segment in enumerate(segment_generator):
                batch_buffer.append(segment)
 
                if len(batch_buffer) == batch_size:
                    batch_tensor = torch.stack(batch_buffer, dim=0)  # [B,NC,C,CS,H,W] uint8 CPU
                    feats = inference(batch_tensor)
                    video_features.append(feats.detach().cpu())
                    batch_buffer.clear()
                    del batch_tensor, feats
                    if segment_idx % log_every == 0:
                        print_info(video_idx, vp, segment_idx, total_segments, segment)
 
            if len(batch_buffer) > 0:
                batch_tensor = torch.stack(batch_buffer, dim=0)
                feats = inference(batch_tensor)
                video_features.append(feats.detach().cpu())
                batch_buffer.clear()
                print_info(video_idx, vp, segment_idx, total_segments, segment)
                del batch_tensor, feats
 
            print()
 
            if len(video_features) > 0:
                video_feature_tensor = torch.cat(video_features, dim=0)
                data = {
                    "feats": video_feature_tensor,
                    "model": core_model.feature_extractor.__class__.__name__.lower(),
                    "clip_size": core_model.clip_size,
                    "overlap": core_model.overlap,
                    "max_segment_size": core_model.max_clip_per_segment,
                    "min_segment_size": core_model.min_clip_per_segment,
                    "fps": fps,
                    "number_of_segment": core_model.number_of_segments,
                    "width": W,
                    "height": H,
                }
                torch.save(data, save_path)
            else:
                raise NO_SEGMENT_ERROR

            if save_debug_video and all_debug_frames:
                full_segment = torch.cat(all_debug_frames, dim=0).to(torch.float32) # [NS, NC, C, CS, H, W]
                NS, NC, C, CS, H, W = full_segment.shape
                full_segment = full_segment.permute(0, 1, 3, 4, 5, 2) # [NS, NC, CS, H, W, C]
                full_segment = (full_segment * 255.0).clamp(0, 255).to(torch.uint8)
                full_segment_np = full_segment.cpu().contiguous().numpy()

                debug_save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(vp))[0]}_debug.mp4")
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(debug_save_path, fourcc, fps, (W, H))

                for segment_idx in range(NS):
                    for clip_idx in range(NC):
                        for frame_idx in range(CS):
                            frame_rgb = full_segment_np[segment_idx, clip_idx, frame_idx]
                            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            cv2.putText(frame_bgr, f"Seg:{segment_idx+1}/{total_segments}\nClip:{clip_idx+1}/{NC}", 
                                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            out.write(frame_bgr)
                            del frame_rgb, frame_bgr

                out.release()
                
                print(f"[DEBUG] Video olarak kaydedildi: {debug_save_path}")
                
                all_debug_frames.clear()
                del full_segment, full_segment_np
 
            del video_feature_tensor, video_features, segment_generator, data
            gc.collect()
            torch.cuda.empty_cache()
 
        except Exception as e:
            with open(skip_log_path, "a", encoding="utf-8") as f:
                f.write(f"{vp}\t{str(e)}\n")
            print(f"\n[Skipped] {vp} -> Err: {str(e)}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
