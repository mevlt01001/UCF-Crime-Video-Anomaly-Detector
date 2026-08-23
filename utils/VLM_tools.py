import os
import cv2
import json
import base64
import multiprocessing as mp
import numpy as np
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from llama_cpp._utils import suppress_stdout_stderr

from .video_preprocess import generate_frames


VIDEO_SEGMENT_MODEL     = "video_segmenter_model.onnx.plan"
QWEN_VLM_MODEL          = "Qwen3VL2B/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
QWEN_VLM_MODEL_MMPROJ   = "Qwen3VL2B/mmproj-Qwen3VL-2B-Instruct-F16.gguf"
GEMMA_VLM_MODEL         = "Gemma4E2B/gemma-4-E2B_q4_0-it.gguf"
GEMMA_VLM_MODEL_MMPROJ  = "LLMs/gemma-4-E2B-it-Q8_0.gguf"

#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#
VLM_SYS_PROMT = "You are an VLM model. Your mission is describing in summary what you see in video."

DATA_FORMAT = {
    "summary": "A forklift accident and risk of injury were observed in the video.",
    "events": [
        {"event_1": "Forklift overturned"},
        {"event_2": "Person lying motionless on the ground"},
        {"event_n": "A group of person running"},
    ],
    "risk": "5",
    "actions": [
        "Call the medical team",
        "Secure the area"
    ]
}

QA_MODEL_SYS_PROMT = (
    "You are an describer model. Your mission is generate formatted output according to event's description comes from QwenVLM model\n"
    "SUMMARIZE: Firstly, generete a summarize for video and each event, both in short, no event means 'There are no risky or at least not detected'\n"
    "EVALUATE to RISK: And then evaluate to risk between 0 to 5 accordig to event's description\n"
    "GENERATE to ACTIONS: Finally generate a best actions for this stuation. (You should think as if you tell what another person should to do in this stuation)\n"
    f"Return an output like {DATA_FORMAT}"
)

class Model_Manager():
    def __init__(self):
        self.messages = []
    
    def load_vlm(self, model_path, clip_model_path, sys_promt, n_gpu_layers=22):
        self.model_path = model_path
        
        self.chat_handler = Qwen25VLChatHandler(
            clip_model_path=clip_model_path, 
            verbose=False
        )
        
        self.model = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,
            n_gpu_layers=n_gpu_layers, 
            n_ctx=6144,                # Keeps text context window safe
            flash_attn=True,           # Crucial memory compression for VRAM
            n_threads=8,               # Optimizes CPU execution if any math spills over
            n_batch=512,               # Lower batch size prevents sudden VRAM spikes
            offload_kqv=True,          # Forces Attention tracking onto the GPU VRAM
            verbose=False,
        )
        self.set_sys_promt(sys_promt)
        
    def load_QA_model(self, model_path, sys_promt, n_gpu_layers=-1):
        self.model_path = model_path
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=4096,
            n_threads=8,
            n_batch=512,
            flash_attn=True,
            verbose=False
        )
        self.set_sys_promt(sys_promt)

    def set_sys_promt(self, promt):
        self.messages = [{"role": "system", "content": promt}]    

    def inference(self, text: str, frames: np.ndarray = None, reset_sys_promt: str = None):
        assert hasattr(self, "model"), "Firstly load a model by using load_QA_model() or load_vlm()"

        if reset_sys_promt is not None:
            self.set_sys_promt(reset_sys_promt)

        content = []

        if frames is not None:
            for i in range(frames.shape[0]):
                frame = frames[i]
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}
                })

        if text:
            content.append({"type": "text", "text": text})

        message = {"role": "user", "content": content}
        self.messages.append(message)

        with suppress_stdout_stderr(disable=False):
            response = self.model.create_chat_completion(
                messages=self.messages,
                max_tokens=1024,
                temperature=0.7,
                repeat_penalty=1.05,
                frequency_penalty=0.2,
                top_p=0.95
            )

        print(response), exit()

        answer = response["choices"][0]["message"]["content"].strip()

        self.messages.append({"role": "assistant", "content": answer})
        return answer


def _worker(model_type, load_kwargs, task_q, result_q):
    """Subprocess içinde çalışacak ve CUDA context'i izole edecek worker."""
    mm = Model_Manager()
    
    if model_type == "vlm":
        mm.load_vlm(**load_kwargs)
    elif model_type == "qa":
        mm.load_QA_model(**load_kwargs)

    while True:
        task = task_q.get()
        if task is None:
            if hasattr(mm, "model"):
                mm.model.close()
            break
            
        text, frames, reset_sys_promt = task
        try:
            res = mm.inference(text, frames, reset_sys_promt)
            result_q.put({"status": "success", "data": res})
        except Exception as e:
            result_q.put({"status": "error", "error": str(e)})

class ModelProcess:
    def __init__(self, model_type, load_kwargs):
        ctx = mp.get_context("spawn") 
        self.task_q = ctx.Queue()
        self.result_q = ctx.Queue()
        self.proc = ctx.Process(
            target=_worker,
            args=(model_type, load_kwargs, self.task_q, self.result_q)
        )
        self.proc.start()

    def inference(self, text, frames=None, reset_sys_promt=None):
        self.task_q.put((text, frames, reset_sys_promt))
        result = self.result_q.get()
        if result["status"] == "error":
            raise RuntimeError(f"Subprocess Error: {result['error']}")
        return result["data"]

    def shutdown(self):
        self.task_q.put(None)
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join()

def vlm_infernce(vlm_path:os.PathLike,
                 vlm_mmproj_path:os.PathLike,
                 vlm_sys_promt:os.PathLike,
                 vlm_promt: str,
                 segments:list[dict], 
                 video_path:os.PathLike):

    load_args = {
        "model_path": vlm_path,
        "clip_model_path": vlm_mmproj_path,
        "sys_promt": vlm_sys_promt
    }
    vlm_proc = ModelProcess(model_type="vlm", load_kwargs=load_args)

    responses = []
    for segment in segments:
        frames, start_sec, end_sec = generate_frames(
            video_path,
            segment["start_time"],
            segment["end_time"],
            all_video=False,
            FPS=5,
            max_frames=32
        )
        print_run_info(vlm_path, video_path, start_sec, end_sec)
        
        response = vlm_proc.inference(vlm_promt, frames, reset_sys_promt=vlm_sys_promt)
        responses.append(response)
        
    # Summary
    frames, start_sec, end_sec = generate_frames(video_path,0,0,all_video=True, FPS=5, max_frames=32)
    print_run_info(vlm_path, video_path, start_sec, end_sec)
    
    summary = vlm_proc.inference(
        "Just summarize this video.", 
        frames, 
        reset_sys_promt=vlm_sys_promt
    )

    vlm_proc.shutdown() 

    vlm_events_dict = {}
    for i, resp in enumerate(responses):
        vlm_events_dict[f"Event {i+1}"] = resp

    return {"Events": vlm_events_dict, "Summary": summary}


def qa_inference(qa_path:os.PathLike,
                 qa_sys_promt:os.PathLike,
                 qa_promt: str):
    
    # Modeli ayrı process'te başlat
    load_args = {
        "model_path": qa_path,
        "sys_promt": qa_sys_promt
    }
    qa_proc = ModelProcess(model_type="qa", load_kwargs=load_args)
    
    response = qa_proc.inference(qa_promt, frames=None)
    
    qa_proc.shutdown()
    
    return response


# ---------------- Yardımcı Fonksiyonlar ----------------

def print_run_info(model_path, video_path, start_sec, end_sec):
    model_name = os.path.basename(model_path)
    video_name = os.path.basename(video_path)
    start_sec  = seconds_to_mmss(start_sec)
    end_sec    = seconds_to_mmss(end_sec)
    print(f"{model_name} running between on {start_sec} - {end_sec} of video {video_name}")

def seconds_to_mmss(seconds:float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"

def save_json_data(qa_response:dict, segments:list[dict], save_dir:os.PathLike, video_path:str):
    qa_response = json.loads(qa_response)
    basename = os.path.basename(video_path)
    save_root = os.path.join(save_dir, os.path.splitext(basename)[0])
    
    os.makedirs(save_root, exist_ok=True)
    
    risk_map = {
        "0": "Risk Yok", "1": "Çok Düşük Risk", "2": "Düşük Risk",
        "3": "Orta Risk", "4": "Yüksek Risk", "5": "Çok Yüksek Risk"
    }

    data = {
        "summary": qa_response.get("summary", ""),
        "events": [],
        "risk": risk_map.get(str(qa_response.get("risk", "0")), "Bilinmeyen Risk"),
        "actions": qa_response.get("actions", []),
    }

    for segment, event_val in zip(segments, qa_response.get("events", [])):
        event_str = list(event_val.values())[0] if isinstance(event_val, dict) else event_val
        
        event_entry = {
            "time_stamp": [seconds_to_mmss(segment["start_time"]), seconds_to_mmss(segment["end_time"])],
            "event": event_str,
            "clip": segment.get("clip", "")
        }
        data["events"].append(event_entry)

    with open(os.path.join(save_root, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)    
    return data