import gc
import os
import cv2
import torch
import torchvision.io as io
import torch.nn.functional as F

from tqdm import tqdm
from decord import VideoReader, cpu
from typing import Generator, Optional
from torchvision.models.video import *
from .SegmentRankingModel import SegmentRankingModel

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

STR_MODEL_SPECIFY_ERROR = lambda model_name : f"Specified model {model_name} is unavailable. \
Please use fallowing one: {model_creaters.keys()}"

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
                 frames_per_clip:int = 16,
                 overlap:int = 0,
                 clips_per_segment:int = 8,
                 enable_feature_extractor:bool = True,
                 enable_anomaly_classifier:bool = True
                 ):
        super().__init__()
        self.frames_per_clip = frames_per_clip
        self.overlap = overlap
        self.stride = frames_per_clip - overlap
        self.clips_per_segment = clips_per_segment
        self.enable_feature_extractor = enable_feature_extractor
        self.enable_anomaly_classifier = enable_anomaly_classifier
        self.__set_video_classifier(feature_extractor_model)
        self.__set_classifier_feature_dim()
        self.segment_ranker_model = SegmentRankingModel(self.feature_dim)

    @staticmethod
    def clip_generator(video_path:os.PathLike, 
                       frames_per_clip:int, 
                       overlap:int,
                       fps:int = 30,
                       width:int = 224,
                       height:int = 224) -> Generator[torch.Tensor, None, None]:
        """
        Generate clips from a video file.

        :param os.PathLike video_path: Path to the video file.
        :param int frames_per_clip: Number of frames in a single clip.
        :param int overlap: Number of overlapping frames between consecutive clips.
        :param int fps: (Preprocess) Frame per Seconf for video.
        :param width,height int: (Preprocess) Video frame size.
        :return torch.Tensor: (Generator, Yields) A tensor of shape [C, frames_per_clip, H, W] representing a clip.
        """
        vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
        org_fps = vr.get_avg_fps()
        frame_indices = torch.arange(0, len(vr), step=org_fps/fps).long()
        total_frames = len(frame_indices)
        stride = frames_per_clip - overlap

        if total_frames > fps * 60 * 10:
            raise MemoryError(f"Video is too long! ({total_frames} frames)")

        for start_idx in range(0, total_frames - frames_per_clip + 1, stride):
            end_idx = start_idx + frames_per_clip
            if end_idx > total_frames:
                break
            clip_indices = frame_indices[start_idx:end_idx]
            clip_frames = vr.get_batch(clip_indices).asnumpy()  # Shape: [frames_per_clip, H, W, C]
            clip_tensor = torch.from_numpy(clip_frames).permute(3, 0, 1, 2)  # Shape: [C, frames_per_clip, H, W]
            clip_tensor = clip_tensor.float()/255.0
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
        This method used to generate segments from given video.

        :param video_path os.PathLike: Video path.
        :param clips_per_segment int: Number of clips per segment.
        :param frames_per_clip int: Number of frames per clip.
        :param overlap int: Number of overlapping frames intersected clips.
        :param fps int: (Preprocess) Video's FPS
        :param width int: Video width
        :param height int: Video height
        """

        clip_generator = Video_Feature_Extractor.clip_generator(video_path, frames_per_clip, overlap, fps, width, height)
        clips_buffer = []

        for clip in clip_generator:
            if clip is None: break

            clips_buffer.append(clip)
            if len(clips_buffer) == clips_per_segment:
                segment_tensor = torch.stack(clips_buffer, dim=0)
                clips_buffer.clear()
                yield segment_tensor

        if clips_buffer:
            segment_tensor = torch.stack(clips_buffer, dim=0)
            pad = clips_per_segment - segment_tensor.shape[0]
            if pad < (clips_per_segment / 2):
                segment_tensor = F.pad(segment_tensor, pad=(0, 0, 0, 0, 0, 0, 0, 0, 0, pad))
                yield segment_tensor
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.

        :param torch.Tensor x: Input tensor of shape [B, C, S, H, W], where B is batch size, C is number of channels, S is number of frames, H is height, and W is width.
        :return torch.Tensor: Output tensor of shape [B, D], where D is the feature dimension.
        """
        is_batched = x.dim() == 6
        
        if is_batched:
            B, Clips, C, S, H, W = x.shape
            x = x.view(B * Clips, C, S, H, W)
            
        if self.enable_feature_extractor:
            x = self.feature_extractor(x)       # -> [B * Clips, Dim] or [Clips, Dim]
            if is_batched:
                x = x.view(B, Clips, -1).mean(dim=1) 
            else:
                x = x.mean(dim=0)               # -> [Dim]

        if self.enable_anomaly_classifier:
            x = self.segment_ranker_model(x)    # -> [B, 1] veya [1]
            
        return x

    def trt_inference(self, segment:torch.Tensor):
        segment = segment.contiguous().to(torch.float32)
        
        self.trt_context.set_input_shape("input", tuple(segment.shape))

        output_shape = tuple(self.trt_context.get_tensor_shape("output"))
        output = torch.empty(output_shape, dtype=torch.float32, device=segment.device)

        self.trt_context.set_tensor_address("input", int(segment.data_ptr()))
        self.trt_context.set_tensor_address("output", int(output.data_ptr()))
        
        stream = torch.cuda.current_stream().cuda_stream
        self.trt_context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()
        
        return output

    def export_onnx(self, filename, imgsz):
        import onnx, onnxsim

        B = self.clips_per_segment
        S = self.frames_per_clip
        H = W = imgsz
        dummy = torch.rand(B,3,S, H, W, device="cpu")

        torch.onnx.export(
            self.to("cpu"),
            dummy,
            filename,
            input_names=["input"],
            output_names=["output"],
            opset_version=19,
        )

        onnx_model = onnx.load(filename)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx_model, check = onnxsim.simplify(onnx_model, check_n=3, test_input_shapes={"input": [B, 3, S, H, W]})
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx.save(onnx_model, filename)

    def export_trt(self, file_name, imgsz):
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
            self.load_trt(trt_filename, trt.Runtime(TRT_LOGGER))
            return

        self.export_onnx(onnx_filename, imgsz)

        if not PARSER.parse_from_file(onnx_filename):
            for error in range(PARSER.num_errors):
                print(PARSER.get_error(error))
            raise RuntimeError(f"ONNX Parsing error for model: {onnx_filename}")


        B = self.clips_per_segment
        S = self.frames_per_clip
        H = W = imgsz
        shape = (B, 3, S, H, W)
        
        OPT_PROFILE.set_shape("input", shape, shape, shape)
        CONFIG.add_optimization_profile(OPT_PROFILE)

        engine_bytes = BUILDER.build_serialized_network(NETWORK, CONFIG)
        
        if engine_bytes is None:
            raise RuntimeError("TRT Plan generation failed!")

        with open(trt_filename, "wb") as f:
            f.write(engine_bytes)

        self.load_trt(trt_filename, trt.Runtime(TRT_LOGGER))
            
    def load_trt(self, trt_path, runtime):
        
        with open(trt_path, 'rb') as f:
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
        self.trt_context = self.trt_engine.create_execution_context()

        if hasattr(self, 'feature_extractor'):
            del self.feature_extractor
            del self.enable_anomaly_classifier
            torch.cuda.empty_cache()

    def __set_video_classifier(self, video_classifier):
        if isinstance(video_classifier, str): 
            # If specified model by name in formed str
            if video_classifier.lower() in model_creaters.keys():
                # If specified model name available
                self.feature_extractor = model_creaters[video_classifier](weights=model_weights[video_classifier].DEFAULT)
            else: 
                # If specified model name does not available
                raise ValueError(STR_MODEL_SPECIFY_ERROR(video_classifier))
        elif isinstance (video_classifier, (MViT | VideoResNet | S3D | SwinTransformer3d)):
            # If Specified model formed its class form
            self.feature_extractor = video_classifier
        else:
            # If specified model does not met any condition:
             raise ValueError(STR_MODEL_SPECIFY_ERROR(video_classifier))
        
    def __set_classifier_feature_dim(self):

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
            raise ValueError("The feature extractor model does not have a classifier or head attribute.")

def calc_vram_usage_in_mb(model: torch.nn.Module, tensor: torch.Tensor) -> float:

    device = "cuda"
    tensor = tensor.to(device, non_blocking=True)
    
    torch.cuda.reset_peak_memory_stats(device)
    initial_mem = torch.cuda.memory_allocated(device)
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        _ = model(tensor)
        
    peak_mem = torch.cuda.max_memory_allocated(device)
    vram_bytes = peak_mem - initial_mem
    
    del tensor, _
    torch.cuda.empty_cache()
    
    return vram_bytes / (1024 * 1024)


@torch.no_grad()
def extract_feats(extractor: Video_Feature_Extractor,
                  video_paths: list,
                  save_dir: str,
                  batch_size: int = 4,  # Yeni eklenen Batch parametresi
                  imgsz: int = 224,
                  fps: int = 30,
                  skip_log_path: str = None):
    
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
    
    for video_idx, vp in enumerate(video_paths):
        save_path = os.path.join(save_dir, os.path.basename(vp) + ".pt")
        
        if os.path.exists(save_path):
            print(f"Skipping.. (Already exists): {video_idx+1:04d}/{len_videos} -> {os.path.basename(vp)}", flush=True)
            continue

        video_features = [] 
        batch_buffer = []
        
        try:
            segment_generator = Video_Feature_Extractor.segment_generator(
                video_path=vp,
                clips_per_segment=core_model.clips_per_segment,
                frames_per_clip=core_model.frames_per_clip,
                overlap=core_model.overlap,
                fps=fps,
                width=imgsz,
                height=imgsz
            )

            for segment_idx, segment in enumerate(segment_generator):
                batch_buffer.append(segment)

                if len(batch_buffer) == batch_size:
                    batch_tensor = torch.stack(batch_buffer, dim=0).to("cuda", non_blocking=True)
                    
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        feats = extractor(batch_tensor)  
                    
                    video_features.append(feats.detach().cpu().clone())
                    batch_buffer.clear()
                    
                    print(f"Video: {video_idx+1:04d}/{len_videos} | Video: {os.path.basename(vp)} | Segment (Max): {segment_idx+1}", end='\r', flush=True)

            if len(batch_buffer) > 0:
                batch_tensor = torch.stack(batch_buffer, dim=0).to("cuda", non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    feats = extractor(batch_tensor)
                    
                video_features.append(feats.detach().cpu().clone())
                batch_buffer.clear()
                print(f"Video: {video_idx+1:04d}/{len_videos} | Video: {os.path.basename(vp)} | Segment (Max): {segment_idx+1}", end='\r', flush=True)

            print()

            if len(video_features) > 0:
                video_feature_tensor = torch.cat(video_features, dim=0)  
                torch.save(video_feature_tensor, save_path)
            else:
                raise ValueError("There is no segment in this video.")

            del video_feature_tensor, video_features, segment_generator
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            with open(skip_log_path, "a", encoding="utf-8") as f:
                f.write(f"{vp}\t{str(e)}\n")
            print(f"\n[Skipped] {vp} -> Err: {str(e)}", flush=True)