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

        self.video_preprocess = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))], p=0.15),
            AddGaussianNoise(mean=0.0, std=0.03, p=0.15),
            AddSaltAndPepperNoise(amount=0.01, p=0.10),
        ])

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
        
        debug_augmented_clips = [] # n X [NS, chunk_size, C, CS, h, W]

        for start in range(0, NC, chunk_size):
            end = min(start + chunk_size, NC)
            chunk = x[:, start:end]  # [B, nc, C, CS, H, W] uint8
            b, nc = chunk.shape[0], chunk.shape[1]
            
            chunk = chunk.reshape(b * nc, C, CS, H, W)
            chunk = chunk.to(device, non_blocking=True)
            chunk = chunk.float().div_(255.0)
            
            if augmentation is not None:
                n = chunk.shape[0]
                chunk = chunk.permute(0, 2, 1, 3, 4).reshape(n * CS, C, H, W)
                chunk = augmentation(chunk)
                chunk = chunk.reshape(n, CS, C, H, W).permute(0, 2, 1, 3, 4).contiguous() 

            if save_video:
                chunk_debug = chunk.reshape(b, nc, C, CS, H, W)
                debug_augmented_clips.append(chunk_debug.clone().cpu())
                
            feats = self.feature_extractor(chunk)  # [b*nc, Dim, ...]
            feats = feats.view(b, nc, -1).float().sum(dim=1)
            feat_sum = feats if feat_sum is None else feat_sum + feats
            
            del chunk, feats

        x = feat_sum / NC
        if debug_augmented_clips:
            debug_augmented_clips = torch.concat(debug_augmented_clips, dim=1) # [NS, NC, C, CS, H, W]

        return x, debug_augmented_clips

    def fc_layers_forward(self, x):
        x = self.segment_ranker_model(x)  # [B, 1]
        return x
 
    def forward(self, x: torch.Tensor, augmentation: v2.Compose = None, save_debug_clip:bool=False) -> torch.Tensor:
        """
        Forward pass.
        """
        if hasattr(self, "trt_engine"):
            return self.trt_inference(x)
        debug_frames = []
        if self.enable_feature_extractor:
            x, debug_frames = self.feature_extractor_forward(x, augmentation=augmentation, save_video=save_debug_clip)
        if self.enable_fc_layers:
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
                score = self.forward(mini_batches)
            else:
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    score = self.forward(mini_batches)
            return score

        device = next(self.parameters()).device.type
        if hasattr(self, "trt_engine") and self.trt_engine is not None:
            device = "cuda"


        segment_generator, number_of_segments = Video_Feature_Extractor.segment_generator(
            video_path, self.clip_size, self.stride, self.max_clip_per_segment, self.min_clip_per_segment,
            self.number_of_segments, FPS, width, height, None
        )

        total_segments = number_of_segments
        scores = [] # N x [B, 1]
        mini_batches = [] # N x [C, T, H, W]
        
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

    def export_onnx(self, filename:str, imgsz:tuple|int, batch_size:int, dummy_t:int=16):
        """
        Added `dummy_t` parameter and set Axis 2 to dynamic "T".
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        import onnx, onnxsim

        B = batch_size
        DIM = self.feature_dim
        H, W = imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)

        # Dynamic T shape initialization
        dummy = torch.rand(B, 3, dummy_t, H, W, device="cpu")

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
            dynamic_shapes = {'input': {0: 'B', 2: 'T'}} if self.enable_feature_extractor else {'input': {0: 'B'}},
            opset_version=19,
            do_constant_folding=True,
        )

        onnx_model = onnx.load(filename)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        
        if self.enable_feature_extractor:
            onnx_model, check = onnxsim.simplify(onnx_model, test_input_shapes={"input": [B, 3, dummy_t, H, W]})
        else:
            onnx_model, check = onnxsim.simplify(onnx_model, test_input_shapes={"input": [B, DIM]})
            
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx.save(onnx_model, filename)

        return B, H, W, DIM

    def export_trt(self, file_name, imgsz, batch_size, min_t=4, max_t=1024):
        """
        TensorRT expects bounds for dynamic spatial/temporal sizes. 
        min_t and max_t sets lower/upper bounds for temporal chunks.
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
            
        B, H, W, DIM = self.export_onnx(onnx_filename, imgsz, batch_size)

        if not PARSER.parse_from_file(onnx_filename):
            for error in range(PARSER.num_errors):
                print(PARSER.get_error(error))
            raise ONNX_PARSING_ERROR(onnx_filename)

        if not self.enable_feature_extractor and self.enable_fc_layers:
            min_shape = (1, DIM)
            max_shape = (B, DIM)
            OPT_PROFILE.set_shape("input", min_shape, max_shape, max_shape)
        elif self.enable_feature_extractor:
            min_shape = (1, 3, min_t, H, W)
            opt_shape = (B, 3, 16, H, W)
            max_shape = (B, 3, max_t, H, W)
            OPT_PROFILE.set_shape("input", min_shape, opt_shape, max_shape)
        else:
            raise EMPTY_MODEL_EXPORT_ERROR(self.enable_feature_extractor,
                                            self.enable_fc_layers)
        
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
        return feats, debug_frames
 
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
                    feats, debug_frames = inference(batch_tensor)
                    video_features.append(feats.detach().cpu())
                    batch_buffer.clear()
                    del batch_tensor, feats
                    if segment_idx % log_every == 0:
                        print_info(video_idx, vp, segment_idx, total_segments, segment)
 
            if len(batch_buffer) > 0:
                batch_tensor = torch.stack(batch_buffer, dim=0)
                feats, debug_frames = inference(batch_tensor)
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
