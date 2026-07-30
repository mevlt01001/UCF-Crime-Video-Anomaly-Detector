import gc
import os
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .video_preprocess import fetch_video_patches
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights

class FeatureExtractor(nn.Module):
    def __init__(self, clip_size: int = 16, overlap: int = 0, max_frames: int = 54000):
        super(FeatureExtractor, self).__init__()
        self.clip_size = clip_size
        self.overlap = overlap
        self.stride = clip_size - overlap
        self.backbone = None

        max_clips = (max_frames - clip_size) // self.stride + 1
        self.indices = torch.zeros(max_clips, clip_size, dtype=torch.long)
        for i in range(max_clips):
            self.indices[i] = torch.arange(i * self.stride, i * self.stride + clip_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if hasattr(self, "trt_engine"):
            return self.trt_inference(x)


        B, C, S, H, W = x.shape

        num_clips = (S - self.clip_size) // self.stride + 1
        
        valid_indices = self.indices[:num_clips] 
        flat_indices = valid_indices.flatten()

        x_clips = x.index_select(dim=2, index=flat_indices.to(x.device)) 
        
        x_clips = x_clips.view(B, C, num_clips, self.clip_size, H, W) # [B, C, S, ClipSize, H, W]
        x_clips = x_clips.permute(0, 2, 1, 3, 4, 5) # [B, S, C, ClipSize, H, W]

        h = x_clips.reshape(-1, C, self.clip_size, H, W) # [B*S, C, ClipSize, H, W]
        h = self.backbone(h) # [B*S, Dim]
        h = h.reshape(B, num_clips, -1)
        h = h.mean(dim=1)  

        return h

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

        model_path = filename
        dummy = torch.rand(1,3,60,*imgsz, device="cpu")

        torch.onnx.export(
            self.to("cpu"),
            dummy,
            model_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes = {
                'x': {0: 'B', 2: 'S'},
            },
            opset_version=19,
            do_constant_folding=True,
        )

        onnx_model = onnx.load(model_path)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx_model, check = onnxsim.simplify(onnx_model, check_n=3, test_input_shapes={"input": [1, 3, 60, *imgsz]})
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx.save(onnx_model, model_path)

    def export_trt(self, imgsz):
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        
        BUILDER = trt.Builder(TRT_LOGGER)
        NETWORK = BUILDER.create_network()
        
        PARSER = trt.OnnxParser(NETWORK, TRT_LOGGER)
        
        CONFIG = BUILDER.create_builder_config()
        OPT_PROFILE = BUILDER.create_optimization_profile()

        file_name = "Feature_Extractor"
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


        
        min_shape = (1, 3, self.clip_size, imgsz[1], imgsz[0])
        opt_shape = (1, 3, 30, imgsz[1], imgsz[0])
        max_shape = (1, 3, 235, imgsz[1], imgsz[0])
        
        OPT_PROFILE.set_shape("input", min_shape, opt_shape, max_shape)
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
            del self.segment_ranker
            torch.cuda.empty_cache()


def calc_vram_usage_in_mb(model: nn.Module, tensor: torch.Tensor) -> float:

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
def extract_feats(extractor: FeatureExtractor,
                  video_paths: list,
                  save_dir: str,
                  patch_size: int = 32,
                  resize_dim: tuple = (112, 112),
                  crop_dim: tuple = (112, 112),
                  target_fps: int = 30,
                  safe_vram_mb: float = 3200.0,
                  max_sec: float = 241.5,
                  skip_log_path: str = None):

    os.makedirs(save_dir, exist_ok=True)
    if skip_log_path is None:
        skip_log_path = os.path.join(save_dir, "skipped_videos.txt")

    num_gpus = torch.cuda.device_count()

    extractor.export_trt(crop_dim)

    has_trt = hasattr(extractor, "trt_engine")
    is_dp = False
    if num_gpus > 1 and not has_trt:
        extractor = torch.nn.DataParallel(extractor)
        is_dp = True

    core_model = extractor.module if is_dp else extractor
    clip_size = core_model.clip_size

    if not is_dp and not has_trt:
        extractor = extractor.to("cuda")
        
    extractor.eval()

    len_videos = len(video_paths)
    for video_idx, vp in enumerate(video_paths):
        video_features = []
        save_path = os.path.join(save_dir, os.path.basename(vp) + ".pt")

        cap_temp = cv2.VideoCapture(vp)
        fps_temp = cap_temp.get(cv2.CAP_PROP_FPS)
        frames_temp = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_temp.release()
        
        duration = frames_temp / fps_temp if fps_temp > 0 else 0
        patch_size_K = max(1.0, duration / max_sec)
        len_segments = int(patch_size * patch_size_K)

        if os.path.exists(save_path):
            continue

        try:
            
            segment_generator = fetch_video_patches(vp, target_fps=target_fps, patch_size=patch_size, max_sec=max_sec,
                                                    resize_dim=resize_dim, crop_dim=crop_dim)
            
            if has_trt:
                batch_size = 1
            else:
                dummy = torch.rand(1, 3, clip_size, *crop_dim)
                vram_usage_mb = max(10.0, calc_vram_usage_in_mb(extractor, dummy))
                calculated_batch = int(safe_vram_mb // vram_usage_mb)
                batch_size = max(num_gpus, (calculated_batch // num_gpus) * num_gpus) if is_dp else max(1, calculated_batch)

            batch_buffer = []
            for segment_idx, segment in enumerate(segment_generator):
                segment = segment.float().div_(255.0)
                if segment.shape[2] < clip_size:
                    pad_size = clip_size - segment.shape[2]
                    segment = F.pad(segment, (0, 0, 0, 0, 0, pad_size))
                
                batch_buffer.append(segment)

                if len(batch_buffer) == batch_size or segment_idx == len_segments - 1:
                    batch = torch.cat(batch_buffer, dim=0).to("cuda", non_blocking=True)
                    
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        feats = extractor(batch)  

                    video_features.append(feats.detach().cpu().clone())
                    batch_buffer.clear()

                    del batch, feats
                    torch.cuda.empty_cache()

                print(f"Video: {video_idx+1:04d}/{len_videos} | Batch: {batch_size} | Segment: {segment_idx+1}/{len_segments}", end='\r', flush=True)

            print()

            video_feature = torch.cat(video_features, dim=0)  
            torch.save(video_feature, save_path)

            del video_feature, video_features, segment_generator
            gc.collect()
            
        except Exception as e:
            with open(skip_log_path, "a") as f:
                f.write(f"{vp}\t{e}\n")
            print(f"\n[SKIP] {vp} -> {e}", flush=True)