import cv2
import torch
import numpy as np

def fetch_video_patches(video_path: str, 
                        target_fps: int = 30, 
                        patch_size: int = 32, 
                        resize_dim: tuple = (112, 112)):
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    ratio = target_fps / original_fps

    original_number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_number_of_frames = int(original_number_of_frames * ratio)

    number_of_patch_elements = target_number_of_frames // patch_size
    clean_target_frames = number_of_patch_elements * patch_size
    
    frames = []
    current_orig_idx = -1
    last_frame = None

    for target_idx in range(clean_target_frames):
        needed_orig_idx = int(target_idx / ratio)

        while current_orig_idx < needed_orig_idx:
            ret, frame = cap.read()
            if not ret:
                break
            current_orig_idx += 1
            last_frame = frame
        
        if last_frame is None:
            break
            
        processed_frame = last_frame.copy()
        if resize_dim: 
            processed_frame = cv2.resize(processed_frame, resize_dim)
            
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        frames.append(processed_frame)    
        
        if len(frames) == number_of_patch_elements:
            batch_tensor = process_patch(frames)
            yield batch_tensor
            frames = []

    cap.release()

def process_patch(frames_list):

    np_frames = np.array(frames_list)  # (T, H, W, 3) uint8
    tensor = torch.from_numpy(np_frames)  # uint8
    tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 4, 1, 2, 3).contiguous()  # (1, 3, T, H, W) uint8
    return tensor