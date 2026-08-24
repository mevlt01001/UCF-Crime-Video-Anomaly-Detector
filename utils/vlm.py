import os
import cv2
import base64
import tempfile
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class VLM_Manager:
    def __init__(self, 
                 api_key: str, 
                 base_url: str, 
                 model_name: str = "vlm", 
                 system_prompt: str = None, 
                 temperature: float = 0.0):
        
        self.vlm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            request_timeout=1800
        )
        self.system_prompt = system_prompt
        self.history = []
        self.reset_context()

    def reset_context(self):
        self.history = []
        if self.system_prompt:
            self.history.append(SystemMessage(content=self.system_prompt))

    def _frames_to_mp4_base64(self, frames: np.ndarray, fps: float = 5.0) -> str:

        N, H, W, C = frames.shape
        
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (W, H))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            
            with open(temp_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")
                
            return video_b64
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def run(self, text: str, frames: np.ndarray = None, fps: float = 5.0) -> str:
        content = []
        
        if text:
            content.append({"type": "text", "text": text})
            
        if frames is not None:
            video_b64 = self._frames_to_mp4_base64(frames, fps=fps)
            
            content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
            })

        self.history.append(HumanMessage(content=content))
        
        try:
            response = self.vlm.invoke(self.history)
            self.history.append(AIMessage(content=response.content))
            return response.content
        except Exception as e:
            self.history.pop()
            return f"[VLM HATA]: {str(e)}"

