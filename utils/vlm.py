import os
import cv2
import base64
import tempfile
import numpy as np

from typing import Optional
from openai import OpenAI

from .env import env_first, env_float, env_int, env_require

# EVREN vlm: image_url yasak ("At most 0 image(s)"). video_url zorunlu.
# 1 kare / yüksek çözünürlük Qwen3VLProcessor'ı kırıyor.
VLM_MAX_EDGE = env_int("VLM_MAX_EDGE", 448)
VLM_DIMENSION_ALIGNMENT = env_int("VLM_DIMENSION_ALIGNMENT", 32)
VLM_MAX_FRAMES = env_int("VLM_MAX_FRAMES", 128)
VLM_MIN_FRAMES = env_int("VLM_MIN_FRAMES", 8)
VLM_DEFAULT_ENCODE_FPS = env_float("VLM_DEFAULT_ENCODE_FPS", 5.0)
VLM_MIN_ENCODE_FPS = env_float("VLM_MIN_ENCODE_FPS", 0.25)
VLM_MAX_ENCODE_FPS = env_float("VLM_MAX_ENCODE_FPS", 30.0)
VLM_FORMAT = "video_url-mp4"


class VLM_Manager:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: str = None,
        temperature: float = 0.0,
    ):
        self.model_name = model_name or env_first("EVREN_VLM_MODEL", "VLM_NAME", default="vlm")
        self.temperature = temperature
        self.system_prompt = system_prompt
        base = base_url or env_first("EVREN_BASE_URL", "EVREN_URL")
        if not base:
            raise RuntimeError("EVREN_BASE_URL veya EVREN_URL eksik. .env dosyasını kontrol et.")
        self.client = OpenAI(
            api_key=api_key or env_require("EVREN_API_KEY"),
            base_url=base,
            timeout=1800,
        )
        self.history = []
        self.reset_context()

    def reset_context(self):
        self.history = []
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})

    @staticmethod
    def _resize(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(1.0, VLM_MAX_EDGE / max(h, w))
        nw = max(VLM_DIMENSION_ALIGNMENT, int(round(w * scale / VLM_DIMENSION_ALIGNMENT) * VLM_DIMENSION_ALIGNMENT))
        nh = max(VLM_DIMENSION_ALIGNMENT, int(round(h * scale / VLM_DIMENSION_ALIGNMENT) * VLM_DIMENSION_ALIGNMENT))
        if nw == w and nh == h:
            return frame
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    def _prepare_frames(self, frames: np.ndarray) -> np.ndarray:
        arr = np.asarray(frames)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.shape[0] > VLM_MAX_FRAMES:
            idxs = np.linspace(0, arr.shape[0] - 1, VLM_MAX_FRAMES, dtype=int)
            arr = arr[idxs]
        resized = [self._resize(f) for f in arr]
        while len(resized) < VLM_MIN_FRAMES:
            resized.append(resized[-1])
        return np.stack(resized)

    @staticmethod
    def calculate_encode_fps(frame_count: int, source_duration: float = None) -> float:
        if source_duration is None or source_duration <= 0:
            return VLM_DEFAULT_ENCODE_FPS
        return float(np.clip(frame_count / source_duration, VLM_MIN_ENCODE_FPS, VLM_MAX_ENCODE_FPS))

    def _frames_to_mp4_b64(self, frames: np.ndarray, encode_fps: float) -> str:
        n, h, w, _ = frames.shape
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            writer = None
            for fourcc_name in ("avc1", "mp4v"):
                fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                writer = cv2.VideoWriter(temp_path, fourcc, float(encode_fps), (w, h))
                if writer.isOpened():
                    break
                writer.release()
                writer = None
            if writer is None:
                raise RuntimeError("VideoWriter açılamadı")
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            with open(temp_path, "rb") as f:
                data = f.read()
            if len(data) < 100:
                raise RuntimeError("mp4 boş üretildi")
            return base64.b64encode(data).decode("utf-8")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def run(self, text: str, frames: np.ndarray = None, source_duration: float = None, encode_fps: float = None) -> str:
        content = []
        if text:
            content.append({"type": "text", "text": text})

        sent_shape = None
        n_sent = 0
        actual_encode_fps = None
        if frames is not None:
            prepared = self._prepare_frames(frames)
            n_sent, h, w, _ = prepared.shape
            sent_shape = (h, w, 3)
            actual_encode_fps = encode_fps or self.calculate_encode_fps(n_sent, source_duration)
            video_b64 = self._frames_to_mp4_b64(prepared, actual_encode_fps)
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                }
            )

        self.history.append({"role": "user", "content": content})
        marker_encode_fps = None if actual_encode_fps is None else round(actual_encode_fps, 4)
        marker = (
            f"[{VLM_FORMAT} n={n_sent} shape={sent_shape} "
            f"encode_fps={marker_encode_fps} source_duration={source_duration}]"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                temperature=self.temperature,
            )
            text_out = response.choices[0].message.content or ""
            self.history.append({"role": "assistant", "content": text_out})
            return f"{marker}\n{text_out}"
        except Exception as e:
            self.history.pop()
            msg = str(e)
            if len(msg) > 400:
                msg = msg[:400] + "…"
            return f"{marker}\n[VLM HATA]: {msg}"
