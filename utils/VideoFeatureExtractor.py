import gc
import os
import cv2
import torch
import threading
import torchvision.io as io
import torch.nn.functional as F

from tqdm import tqdm
from queue import Queue
from decord import VideoReader, cpu
from typing import Generator, Optional
from torchvision.models.video import *
from .SegmentRankingModel import SegmentRankingModel
from .video_preprocess import get_video_length, save_segment_clips
from .visualization_tools import plot_anomaly_timeline

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
# LONG_VIDEO_ERROR = lambda frame_count : MemoryError(f"Video is too long! ({frame_count} frames)")
EMPTY_MODEL_EXPORT_ERROR = lambda enable_feature_extractor, enable_fc_layers : AttributeError(f"An \
empty model cannot be exported! (enable_feature_extractor: {enable_feature_extractor}, enable_fc_layers:{enable_fc_layers}))")
ONNX_PARSING_ERROR = lambda filename : RuntimeError(f"ONNX Parsing error for model: {filename}")
TRT_RUNTIME_GENERATOR_FAILURE = RuntimeError("TRT Plan generation failed!")
MODEL_HAS_NO_CLASSIFIER_ERROR = ValueError("The feature extractor model does not have a classifier or head attribute.")
NO_SEGMENT_ERROR = ValueError("There is no segment in this video.")

class Video_Feature_Extractor(torch.nn.Module):
    """
    This module is designed to extract frames by pre-trained video classifier models: [`MViT`, `VideoResNet`, `S3D`, `SwinTransformer3d` from `torchvision.models.video`].

    :param Optional[MViT | VideoResNet | S3D | SwinTransformer3d | str] feature_extractor_model: Pre-trained video classification model from `torchvision.models.video`.
    :param int frames_per_clip: Number of frames in a single clip.
    :param int overlap: Number of overlapping frames between consecutive clips.
    :param int clips_per_segment: Number of clips in a single segment.
    """
    def __init__(self, 
                 feature_extractor_model:Optional[MViT | VideoResNet | S3D | SwinTransformer3d | str], 
                 frames_per_clip:int,
                 overlap:int,
                 clips_per_segment:int,
                 fc_layer_checkpoint = None,
                 enable_feature_extractor:bool = True,
                 enable_fc_layers:bool = True
                 ):
        super().__init__()
        self.frames_per_clip = frames_per_clip
        self.overlap = overlap
        self.stride = frames_per_clip - overlap
        self.clips_per_segment = clips_per_segment
        self.enable_feature_extractor = enable_feature_extractor
        self.enable_fc_layers = enable_fc_layers
        self.__set_video_classifier(feature_extractor_model)
        self.__set_video_classifier_feature_dim()
        self.segment_ranker_model = SegmentRankingModel(self.feature_dim)
        if fc_layer_checkpoint:
            self.segment_ranker_model.load_state_dict(fc_layer_checkpoint)

    @staticmethod
    def clip_generator(video_path:os.PathLike, 
                       frames_per_clip:int, 
                       overlap:int,
                       fps:int = 30,
                       width:int = 224,
                       height:int = 224) -> Generator[torch.Tensor, None, None]:
        """
        Yields clips from given video. Each yield size is [C, CS, H, W] (C:Channel, CS: Clip size (Number of Frame))
        :param video_path os.Pathlike: Video Path
        :param frames_per_clip int: Number of frames per clip.
        :param overlap int: Number of overllapped frame between two clips.
        :param fps int: FPS.
        :param width int: Frame width.
        :param height int: Frame height.
        """
        
        vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height, num_threads=4)
        org_fps = vr.get_avg_fps()
        frame_indices = torch.arange(0, len(vr), step=org_fps/fps).long()
        total_frames = len(frame_indices)
        stride = frames_per_clip - overlap

        for start_idx in range(0, total_frames - frames_per_clip + 1, stride):
            end_idx = start_idx + frames_per_clip
            if end_idx > total_frames:
                break
            
            clip_indices = frame_indices[start_idx:end_idx]
            
            clip_frames = vr.get_batch(clip_indices).asnumpy() # [N, H, W, C]
            clip_tensor = torch.from_numpy(clip_frames).permute(3, 0, 1, 2) # [C, N, H, W]
            
            yield clip_tensor

    @staticmethod
    def segment_generator(video_path:os.PathLike, 
                          clips_per_segment:int,
                          frames_per_clip:int, 
                          overlap:int,
                          fps:int = 30,
                          width:int = 224,
                          height:int = 224) -> Generator[torch.Tensor, None, None]:
        """
        Yields segments from given video. Yield shape is [SS, C, CS, H, W](SS: Segments size, C: Number of Channel, CS: Clip size, Number of frame)

        :param video_path os.PathLike: Video path.
        :param clips_per_segment int: Number of clips per segment.
        :param frames_per_clip int: Number of frames per clip.
        :param overlap int: Number of overlapping frames intersected clips.
        :param fps int: (Preprocess) Video's FPS
        :param width int: Video width
        :param height int: Video height
        """

        # clip_generator yields clips shaped [C, N, H, W]
        clip_generator = Video_Feature_Extractor.clip_generator(video_path, frames_per_clip, overlap, fps, width, height)
        clips_buffer = []

        for clip in clip_generator:
            if clip is None: break

            clips_buffer.append(clip)
            if len(clips_buffer) == clips_per_segment:
                segment_tensor = torch.stack(clips_buffer, dim=0) # [B, C, N, H, W]
                clips_buffer.clear()
                yield segment_tensor

        if clips_buffer:
            segment_tensor = torch.stack(clips_buffer, dim=0)
            pad = clips_per_segment - segment_tensor.shape[0]
            if pad < (clips_per_segment / 2):
                segment_tensor = F.pad(segment_tensor, pad=(0, 0, 0, 0, 0, 0, 0, 0, 0, pad))
                yield segment_tensor

    def feature_extractor_forward(self, x):
        B, Clips, C, S, H, W = x.shape
        x = x.view(B * Clips, C, S, H, W)
        x = self.feature_extractor(x)
        x = x.view(B, Clips, -1).mean(dim=1) # [B, Dim]
        return x

    def fc_layers_forward(self, x):
        x = self.segment_ranker_model(x)     # [B, 1]
        return x
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.

        :param torch.Tensor x: 
            If `enable_feature_extractor` input tensor of shape [B, SS, C, CS, H, W](B: Number of Segments, SS: Segment Size , C: Number of Channel, CS: Clip size (Number of frames))\n
            If only `enable_fc_layers` input tensor of shape [B, D](B: Number of Segments, D: Feature Dim)\n

        :return torch.Tensor: 
            If only `enable_feature_extractor` Output tensor of shape [B, D](B: Number of Segments, D: Feature Dim) \n
            If `enable_fc_layers` Output tensor of shape [B, 1](B: Number of Segments, 1: Anomaly Score)
        """
        if hasattr(self, "trt_engine"):
            return self.trt_inference(x)
        if x.dim() == 6 and self.enable_feature_extractor:
            x = self.feature_extractor_forward(x)
        if self.enable_fc_layers:
            x = self.fc_layers_forward(x)

        return x

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
        
        import math
        from tqdm import tqdm
        from decord import VideoReader, cpu

        def batch_flush(mini_batches):
            # N x [B, C, CS, H, W] -> [N, B, C, CS, H, W]
            mini_batches = torch.stack(mini_batches, dim=0)
            mini_batches = mini_batches.to(device=device, 
                                            dtype=torch.float32, 
                                            non_blocking=True)
            # Normalization
            mini_batches = mini_batches.div_(255.0)
            return mini_batches

        def inference(mini_batches):
            # Preprocess
            mini_batches = batch_flush(mini_batches)
            if self.trt_engine is not None:
                # If TensorRT Engine enabled
                score = self.forward(mini_batches)
            else:
                # Only Pytorch Inference in FP16
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    score = self.forward(mini_batches)
            return score

        device = next(self.parameters()).device.type
        if self.trt_engine is not None:
            device = "cuda"

        vr = VideoReader(video_path, ctx=cpu(0))
        org_fps = vr.get_avg_fps()
        total_orig_frames = len(vr)
        del vr

        target_frame_count = len(torch.arange(0, total_orig_frames, step=org_fps/FPS))
        stride = self.frames_per_clip - self.overlap
        
        if target_frame_count >= self.frames_per_clip:
            total_clips = (target_frame_count - self.frames_per_clip) // stride + 1
        else:
            total_clips = 0
            
        total_segments = math.ceil(total_clips / self.clips_per_segment)

        segment_generator = Video_Feature_Extractor.segment_generator(
            video_path, self.clips_per_segment, 
            self.frames_per_clip, 
            self.overlap, FPS, 
            width, height
        )

        scores = [] # N x [B, 1]
        mini_batches = [] # N x [B, C, CS, H, W]
        
        pbar = tqdm(total=total_segments, desc=f"Processing {os.path.basename(video_path)}")
        
        try:
            for segment in segment_generator:
                mini_batches.append(segment)
                
                if len(mini_batches) == batch_size:
                    score = inference(mini_batches) # [B,1]
                    scores.append(score.detach().float().cpu())
                    
                    pbar.update(batch_size)
                    
                    del score
                    mini_batches.clear()
                    torch.cuda.empty_cache()

            if mini_batches:
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

    def trt_inference(self, segment:torch.Tensor):
        """
        Run inference on TensorRT engine. Throws `AttributeError` if not runned `export_trt()` method.

        :param segment torch.Tensor: Input for trt engine. Shape is [B, SS, C, CS, H, W](B: Number of Segments, SS: Segment Size , C: Number of Channel, CS: Clip size (Number of frames))
        """
        segment = segment.contiguous().to(device="cuda", dtype=torch.float32)
        
        self.trt_context.set_input_shape("input", tuple(segment.shape))

        output_shape = tuple(self.trt_context.get_tensor_shape("output"))
        output = torch.empty(output_shape, dtype=torch.float32, device=segment.device)

        self.trt_context.set_tensor_address("input", int(segment.data_ptr()))
        self.trt_context.set_tensor_address("output", int(output.data_ptr()))
        
        stream = torch.cuda.current_stream().cuda_stream
        self.trt_context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()
        
        return output

    def export_onnx(self, filename:str, imgsz:tuple|int, batch_size:int):
        """
        Export ONNX file its forward.

        :param filename str: ONNX file name which is going to save.
        :param imgsz tuple,int: Image size. IF tuple H, W = imgsz; else H, W = imgsz, imgsz.
        :param batch_size int: Batch Size.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        import onnx, onnxsim

        B = batch_size
        SS = self.clips_per_segment # Segment Size, number of clips per segmetn
        CS = self.frames_per_clip # Clip size, Number of frames per clip
        DIM = self.feature_dim
        H, W = imgsz if isinstance(imgsz, tuple) else imgsz, imgsz

        dummy = torch.rand(B, SS, 3, CS, H, W, device="cpu")

        if not self.enable_feature_extractor and self.enable_fc_layers:
            dummy = torch.rand(B, DIM)
        elif not (self.enable_feature_extractor or self.enable_fc_layers):
            raise EMPTY_MODEL_EXPORT_ERROR(self.enable_feature_extractor,
                                           self.enable_fc_layers)

        torch.onnx.export(
            self.to("cpu"),
            dummy,
            filename,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes = {'x': {0: 'B'}},
            opset_version=19,
            do_constant_folding=True,
        )

        onnx_model = onnx.load(filename)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx_model, check = onnxsim.simplify(onnx_model, test_input_shapes={"input": [B, SS, 3, CS, H, W]})
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx.save(onnx_model, filename)

        return B, SS, CS, H, W, DIM

    def export_trt(self, file_name, imgsz, batch_size):
        """
        Export TensorRT engine plan file.

        :param filename str: Trt plan file name which is going to save without extantion. For example (Model, Not Model.trt or Model.engine or Model.plan)
        :param imgsz tuple,int: Image size. IF tuple H, W = imgsz; else H, W = imgsz, imgsz.
        :param batch_size int: Batch Size.
        """
        self.eval().cpu()
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        
        BUILDER = trt.Builder(TRT_LOGGER)
        NETWORK = BUILDER.create_network()
        
        PARSER = trt.OnnxParser(NETWORK, TRT_LOGGER)
        
        CONFIG = BUILDER.create_builder_config()
        OPT_PROFILE = BUILDER.create_optimization_profile()
        onnx_filename = file_name+".onnx"
        trt_filename = file_name+".plan"

        if(os.path.exists(trt_filename)):
            # If plan file already exists.
            return self.load_trt(trt_filename, trt.Runtime(TRT_LOGGER))
            
        B, SS, CS, H, W, DIM= self.export_onnx(onnx_filename, imgsz, batch_size)

        if not PARSER.parse_from_file(onnx_filename):
            for error in range(PARSER.num_errors):
                print(PARSER.get_error(error))
            raise ONNX_PARSING_ERROR(onnx_filename)

        min_shape = (1, SS, 3, CS, H, W)
        max_shape = (B, SS, 3, CS, H, W)
        if not self.enable_feature_extractor and self.enable_fc_layers:
            min_shape = (1, DIM)
            max_shape = (B, DIM)
        elif not (self.enable_feature_extractor or self.enable_fc_layers):
            raise EMPTY_MODEL_EXPORT_ERROR(self.enable_feature_extractor,
                                            self.enable_fc_layers)
        
        OPT_PROFILE.set_shape("input", min_shape, max_shape, max_shape)
        CONFIG.add_optimization_profile(OPT_PROFILE)

        engine_bytes = BUILDER.build_serialized_network(NETWORK, CONFIG)
        
        if engine_bytes is None:
            raise TRT_RUNTIME_GENERATOR_FAILURE

        with open(trt_filename, "wb") as f:
            f.write(engine_bytes)

        self.load_trt(trt_filename, trt.Runtime(TRT_LOGGER))
            
    def load_trt(self, trt_path:os.PathLike, runtime):
        """
        Loads TensorRT engine file already generated before. 
        """
        with open(trt_path, 'rb') as f:
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
        self.trt_context = self.trt_engine.create_execution_context()

        if hasattr(self, 'feature_extractor'):
            del self.feature_extractor
            del self.enable_fc_layers
            torch.cuda.empty_cache()

    def __set_video_classifier(self, video_classifier:str|MViT|VideoResNet|S3D|SwinTransformer3d):
        """
        Sets `feature_extractor` model. 
        """
        if isinstance(video_classifier, str): 
            # If specified model by name in formed str
            if video_classifier.lower() in model_creaters.keys():
                # If specified model name available
                self.feature_extractor = model_creaters[video_classifier](weights=model_weights[video_classifier].DEFAULT)
            else: 
                # If specified model name does not available
                raise MODEL_SPECIFY_ERROR(video_classifier)
        elif isinstance (video_classifier, (MViT | VideoResNet | S3D | SwinTransformer3d)):
            # If Specified model formed its class form
            self.feature_extractor = video_classifier
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

@torch.no_grad()
def extract_feats(extractor: Video_Feature_Extractor,
                  video_paths: list,
                  save_dir: str,
                  batch_size: int = 4,
                  imgsz: int|tuple = 224,
                  fps: int = 30,
                  skip_log_path: str = None):

    extractor.enable_feature_extractor = True
    extractor.enable_fc_layers = False
    H, W = imgsz if isinstance(imgsz, (tuple, list)) else (imgsz, imgsz)

    os.makedirs(save_dir, exist_ok=True)
    if skip_log_path is None:
        skip_log_path = os.path.join(save_dir, "skipped_videos.txt")

    num_gpus = torch.cuda.device_count()
    is_dp = num_gpus > 1

    if is_dp:
        extractor = torch.nn.DataParallel(extractor)
    
    extractor = extractor.to("cuda")
    extractor.eval()

    core_model = extractor.module if is_dp else extractor
    len_videos = len(video_paths)

    # Video by Video
    for video_idx, vp in enumerate(video_paths):
        total_segments = calc_total_segments(vp, fps, extractor)

        save_path = os.path.join(save_dir, os.path.basename(vp) + ".pt")
        
        if os.path.exists(save_path):
            print(f"Skipping.. (Already exists): {video_idx+1:04d}/{len_videos} -> {os.path.basename(vp)}", flush=True)
            continue

        video_features = [] # N x [B, Dim]
        batch_buffer = []   # B x [SS, C, CS, H, W]
        
        try:
            segment_generator = Video_Feature_Extractor.segment_generator(
                video_path=vp,
                clips_per_segment=core_model.clips_per_segment,
                frames_per_clip=core_model.frames_per_clip,
                overlap=core_model.overlap,
                fps=fps,
                width=W,
                height=H
            )

            for segment_idx, segment in enumerate(segment_generator):
                batch_buffer.append(segment)

                if len(batch_buffer) == batch_size:
                    # B X [SS, C, CS, H, W] -> [B, SS, C, CS, H, W]
                    batch_tensor = torch.stack(batch_buffer, dim=0).to("cuda", non_blocking=True)
                    batch_tensor = batch_tensor.float() / 255.0
                    
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        feats = extractor(batch_tensor) # [B, Dim]
                    
                    video_features.append(feats.detach().cpu().clone())
                    batch_buffer.clear()
                    
                    print(f"Video: {video_idx+1:04d}/{len_videos} | Video: {os.path.basename(vp)} | Segment: {segment_idx+1}/{total_segments}", end='\n', flush=True)

            if len(batch_buffer) > 0:
                # B X [SS, C, CS, H, W] -> [B, SS, C, CS, H, W]
                batch_tensor = torch.stack(batch_buffer, dim=0).to("cuda", non_blocking=True)
                batch_tensor = batch_tensor.float() / 255.0
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    feats = extractor(batch_tensor) # [B, Dim]
                    
                video_features.append(feats.detach().cpu().clone())
                batch_buffer.clear()
                print(f"Video: {video_idx+1:04d}/{len_videos} | Video: {os.path.basename(vp)} | Segment: {segment_idx+1}/{total_segments}", end='\n', flush=True)

            print()

            if len(video_features) > 0:
                # N x [B, Dim] -> [NxB, Dim]
                video_feature_tensor = torch.cat(video_features, dim=0)  
                data = {
                    "feats":video_feature_tensor,
                    "model":extractor.feature_extractor.__class__.__name__.lower(),
                    "frames_per_clip": extractor.frames_per_clip,
                    "clips_per_segment":extractor.clips_per_segment,
                    "overlap":extractor.overlap,
                    "imgsz":(H, W),
                    "fps":fps
                }
                torch.save(data, save_path)
            else:
                raise NO_SEGMENT_ERROR

            del video_feature_tensor, video_features, segment_generator, data
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            with open(skip_log_path, "a", encoding="utf-8") as f:
                f.write(f"{vp}\t{str(e)}\n")
            print(f"\n[Skipped] {vp} -> Err: {str(e)}", flush=True)

def calc_total_segments(vp, fps, extractor):
    vr = VideoReader(vp, ctx=cpu(0), num_threads=4)
    org_fps = vr.get_avg_fps()
    frame_indices = torch.arange(0, len(vr), step=org_fps/fps).long()
    total_frames = len(frame_indices)
    del vr

    stride = extractor.frames_per_clip - extractor.overlap

    stop_idx = total_frames - extractor.frames_per_clip + 1
    if stop_idx > 0:
        total_clips = len(range(0, stop_idx, stride)) 
    else:
        total_clips = 0

    base_segments = total_clips // extractor.clips_per_segment
    remainder_clips = total_clips % extractor.clips_per_segment

    extra_segment = 0
    if remainder_clips > 0:
        pad = extractor.clips_per_segment - remainder_clips
        if pad < (extractor.clips_per_segment / 2.0):
            extra_segment = 1

    total_segments = base_segments + extra_segment
    return total_segments