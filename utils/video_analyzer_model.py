import gc
import os
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

def get_number_of_segments():
    pass

class Video_Analyzer(torch.nn.Module):
    def __init__(self,
                video_classifier_model: Optional[MViT | VideoResNet | S3D | SwinTransformer3d | str],
                clip_size: int,
                overlap: int,
                max_clips_per_segment: int,
                number_of_segments: int,
                fc_layer_checkpoint=None,
                ):

        super().__init__()
        self.clip_size = clip_size
        self.overlap = overlap
        self.stride = clip_size - overlap
        self.number_of_segments = number_of_segments
        self.max_clips_per_segment = max_clips_per_segment

        self.__set_video_classifier(video_classifier_model)
        self.__set_video_classifier_feature_dim()
        self.segment_ranker_model = SegmentRankingModel(self.feature_dim)
        if fc_layer_checkpoint:
            self.segment_ranker_model.load_state_dict(fc_layer_checkpoint)
        self.transforms = model_weights[self.video_classifier_name].DEFAULT.transforms()
        self.transforms.resize_size = self.transforms.crop_size

    @staticmethod
    @torch.no_grad()
    def clip_generator(segment_frames: torch.Tensor,
                       clip_size: int,
                       stride: int) -> torch.Tensor:

        N, H, W, C = segment_frames.shape
        clips = []

        for stop_idx in range(clip_size, N + 1, stride):
            start_idx = stop_idx - clip_size
            clip = segment_frames[start_idx:stop_idx] # [CS, H, W, C]
            clips.append(clip)

        return torch.stack(clips, dim=0) # [NC, CS, H, W, C]

    @staticmethod
    @torch.no_grad()
    def segment_generator(video_path: os.PathLike,
                          clip_size: int,
                          stride: int,
                          number_of_segments: int,
                          max_clips_per_segment: int,
                          fps: int = 30,
                          width: int = 224,
                          height: int = 224) -> Generator[torch.Tensor, None, None]:

        vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height, num_threads=4)
        org_fps = vr.get_avg_fps()
        frame_indices = torch.arange(0, len(vr), step=org_fps / fps).long()
        total_frames = len(frame_indices)

        frames_per_segment = total_frames//number_of_segments
        max_frames_per_segment = clip_size + (max_clips_per_segment-1)*stride

        k = max(1,frames_per_segment/max_frames_per_segment)
        number_of_segments = int(number_of_segments*k) + 1

        frames_per_segment = total_frames//number_of_segments

        def _generator(video_reader):
            try:
                for i in range(number_of_segments):
                    start_idx = i * frames_per_segment
                    end_idx = start_idx + frames_per_segment
                    segment_indices = frame_indices[start_idx:end_idx] 
                    segment_frames = video_reader.get_batch(segment_indices.tolist()).asnumpy()

                    segment_tensor = torch.from_numpy(segment_frames) # [N, H, W, C]
                    segment_tensor = Video_Analyzer.clip_generator(segment_tensor, clip_size, stride) # [NC, CS, H, W, C]
                    segment_tensor = segment_tensor.permute(0, 1, 4, 2, 3).contiguous() # [NC, CS, C, H, W]

                    yield segment_tensor
            finally:
                del video_reader

        return _generator(vr), number_of_segments

    def feature_extractor_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [NC, C, CS, H, W]
        x = self.feature_extractor(x) # [NC, dim]
        x = x.mean(0, keepdim=True) # [1, dim]
        return x

    def fc_layers_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, dim]
        x = self.segment_ranker_model(x) # [N, 1]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "trt_context"):
            return self.trt_forward(x)
        x = self.feature_extractor_forward(x) # [1, dim]
        x = self.fc_layers_forward(x) # [1, 1]
        return x

    @torch.no_grad()
    def analyze(self, 
                video_path:str, 
                width:int,
                height:int, 
                fps:float,
                threshold=0.3,
                tolerance_sec=3.0,
                padding_sec=2.0,
                save_graph=True,
                save_clips=True,
                save_dir="Video_Analyses"
                ):

        is_trt = hasattr(self, "trt_context") and self.trt_context is not None

        if is_trt:
            device = "cuda"
        else:
            device = next(self.parameters()).device.type if len(list(self.parameters())) > 0 else "cuda"

        segment_generator, total_segments = Video_Analyzer.segment_generator(
            video_path=video_path, 
            clip_size=self.clip_size, 
            stride=self.stride, 
            number_of_segments=self.number_of_segments, 
            max_clips_per_segment=self.max_clips_per_segment,
            fps=fps, 
            width=width, 
            height=height
        )

        scores = [] # N x [1]
        pbar = tqdm(total=total_segments, desc=f"Processing {os.path.basename(video_path)}")
        
        try:
            for segment in segment_generator:
                segment = segment.to(device) # segment.shape [NC, CS, C, H, W]
                segment = self.transforms(segment) # [NC, CS, C, H, W] -> [NC, C, CS, H, W]
                scores.append(self.forward(segment).detach().float().cpu())
                pbar.update(1)
                
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nGot an error [{video_path}]: {e}")
            raise e # Hatayı yutmak yerine programı durdurması için fırlatıyoruz.
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

    @torch.no_grad()
    def trt_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "trt_context") or self.trt_context is None:
            raise RuntimeError("TensorRT context bulunamadı! Önce export_trt() veya load_trt() çalıştırın.")

        if x.device.type != "cuda":
            x = x.to("cuda", non_blocking=True)
        if x.dtype != torch.uint8:
            x = x.to(torch.uint8)
        x = x.contiguous()

        NC, CS, C, H, W = x.shape
        stream = torch.cuda.current_stream()

        if hasattr(self.trt_context, "set_tensor_address"):
            self.trt_context.set_input_shape("video_segment", (NC, CS, C, H, W))
            output = torch.empty((1,), dtype=torch.float32, device="cuda")
            self.trt_context.set_tensor_address("video_segment", x.data_ptr())
            self.trt_context.set_tensor_address("score", output.data_ptr())
            self.trt_context.execute_async_v3(stream_handle=stream.cuda_stream)

        return output
    
    def export_onnx(self, file_path: str, imgsz: int | tuple[int, int] = 224) -> tuple:
        import onnx
        import onnxsim

        self.eval().cpu()
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        CS = self.clip_size
        H, W = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        
        shape = [self.max_clips_per_segment, CS, 3, H, W]
        
        dummy_input = torch.randint(0, 256, shape, dtype=torch.uint8, device="cpu")
        dummy_input = self.transforms(dummy_input)

        torch.onnx.export(
            self,
            dummy_input,
            file_path,
            input_names=["video_segment"],
            output_names=["score"],
            dynamic_axes={'video_segment': {0: 'num_clips'}},
            opset_version=19,
            do_constant_folding=True,
        )

        onnx_model = onnx.load(file_path)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx_model, check = onnxsim.simplify(onnx_model, test_input_shapes={"video_segment": [1, 3, CS, H, W]})
        if check:
            onnx.save(onnx_model, file_path)

        return shape

    def export_trt(self, imgsz: int | tuple[int, int] = 224, fp16: bool = True):
        import tensorrt as trt

        self.eval().cpu()
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        path = "optimized_models/model"
        trt_file_path = path + ".plan"
        onnx_file_path = path + ".onnx"

        if os.path.exists(trt_file_path):
            return self.load_trt(trt_file_path, trt.Runtime(TRT_LOGGER))
            
        _, CS, _, H, W = self.export_onnx(onnx_file_path, imgsz)

        min_shape = (1, 3, CS, H, W)
        opt_shape = (max(1, self.max_clips_per_segment // 2), 3, CS, H, W)
        max_shape = (self.max_clips_per_segment, 3, CS, H, W)

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

        build_engine(onnx_file_path, trt_file_path)
        self.load_trt(trt_file_path, trt.Runtime(TRT_LOGGER))

    def load_trt(self, file_name: os.PathLike, runtime):
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
def extract_feats(analyzer: Video_Analyzer,
                  video_paths: list,
                  save_dir: str,
                  fps: int,
                  save_debug_video: bool,
                  augmentation:v2.Compose=None):
    extractor = analyzer.feature_extractor

    imgsz = analyzer.transforms.crop_size
    H, W = imgsz if isinstance(imgsz, (tuple, list)) else (imgsz, imgsz)

    skip_log_path = os.path.join(save_dir, "skipped_videos.txt")
    os.makedirs(save_dir, exist_ok=True)
 
    num_gpus = torch.cuda.device_count()
    is_dp = num_gpus > 1
    
    if is_dp: 
        extractor = torch.nn.DataParallel(extractor)
 
    extractor = extractor.to("cuda")
    extractor.eval()
 
    len_videos = len(video_paths)
 
    def print_info(video_idx, vp, segment_idx, total_segments, segment):
        info = (
            f"Video: {video_idx + 1:04d}/{len_videos} [{os.path.basename(vp)}] | "
            f"Segment: {segment_idx + 1:03d}/{total_segments} - shape: {segment.shape} | "
        )
        print(info, end='\n', flush=True)
 
    for video_idx, vp in enumerate(video_paths):
        save_path = os.path.join(save_dir, os.path.basename(vp) + ".pt")
        video_features = [] # n [1, Dim]
        debug_writer = None
        
        if save_debug_video:
            debug_save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(vp))[0]}_debug.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            debug_writer = cv2.VideoWriter(debug_save_path, fourcc, fps, (W, H))
 
        try:
            segment_generator, total_segments = Video_Analyzer.segment_generator(
                video_path=vp, clip_size=analyzer.clip_size,
                stride=analyzer.stride,
                number_of_segments=analyzer.number_of_segments,
                max_clips_per_segment=analyzer.max_clips_per_segment, 
                fps=fps, width=W, height=H
            )

            for segment_idx, segment in enumerate(segment_generator):
                NC, CS, C, H, W = segment.shape
                if augmentation:
                    segment = segment.view(-1, C, H, W) # [NC, CS, C, H, W] -> [N, C, H, W]
                    segment = augmentation(segment) # [N, C, H, W]
                    segment = segment.view(NC, CS, C, H, W) # [N, C, H, W] -> [NC, CS, C, H, W]
                    
                if save_debug_video and debug_writer is not None:
                    seg_np = segment.permute(0, 1, 3, 4, 2).numpy() # [NC, CS, H, W, C]
                    
                    for clip_idx in range(NC):
                        for frame_idx in range(CS):
                            frame_bgr = cv2.cvtColor(seg_np[clip_idx, frame_idx], cv2.COLOR_RGB2BGR)
                            cv2.putText(frame_bgr, f"Seg:{segment_idx+1}/{total_segments}\nClip:{clip_idx+1}/{NC}", 
                                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            debug_writer.write(frame_bgr)

                segment = segment.cuda()
                segment = analyzer.transforms(segment)
                feats = extractor(segment).mean(0, keepdim=True) # [1, Dim]
                video_features.append(feats.detach().cpu())
                print_info(video_idx, vp, segment_idx, total_segments, segment)
            print()
 
            if len(video_features) > 0:
                video_feature_tensor = torch.cat(video_features, dim=0) # [N, Dim]
                
                data = {
                    "feats": video_feature_tensor,
                    "model": analyzer.video_classifier_name,
                    "clip_size": analyzer.clip_size,
                    "overlap": analyzer.overlap,
                    "max_clips_per_segment": analyzer.max_clips_per_segment,
                    "fps": fps,
                    "number_of_segments": analyzer.number_of_segments,
                    "width": W,
                    "height": H,
                    "augmentation": augmentation.state_dict() if augmentation is not None else None
                }
                torch.save(data, save_path)
            else:
                raise NO_SEGMENT_ERROR
            
            if save_debug_video and debug_writer is not None:
                debug_writer.release()
                print(f"[DEBUG] Video olarak kaydedildi: {debug_save_path}")
 
            del video_feature_tensor, video_features, segment_generator, data
            gc.collect()
            torch.cuda.empty_cache()
 
        except Exception as e:
            if save_debug_video and debug_writer is not None:
                debug_writer.release()
                
            with open(skip_log_path, "a", encoding="utf-8") as f:
                f.write(f"{vp}\t{str(e)}\n")
            print(f"\n[Skipped] {vp} -> Err: {str(e)}", flush=True)
            
            gc.collect()
            torch.cuda.empty_cache()
