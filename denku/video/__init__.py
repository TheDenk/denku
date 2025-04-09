"""Video processing utilities for the denku package."""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


def get_capture_info(cap: cv2.VideoCapture) -> Tuple[int, int, int, int]:
    """Get video capture information.
    
    Args:
        cap (cv2.VideoCapture): Video capture object
        
    Returns:
        Tuple[int, int, int, int]: Height, width, fps, frame count
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return height, width, fps, frame_count


def read_video(video_path: str, start_frame: int = 0,
               frames_count: Optional[int] = None,
               max_side: Optional[int] = None) -> List[np.ndarray]:
    """Read video frames.
    
    Args:
        video_path (str): Path to video file
        start_frame (int, optional): Starting frame index. Defaults to 0.
        frames_count (Optional[int], optional): Number of frames to read. Defaults to None.
        max_side (Optional[int], optional): Maximum side length. Defaults to None.
        
    Returns:
        List[np.ndarray]: List of video frames
    """
    cap = cv2.VideoCapture(video_path)
    height, width, fps, total_frames = get_capture_info(cap)
    
    if frames_count is None:
        frames_count = total_frames - start_frame
    
    if max_side is not None:
        scale = max_side / max(height, width)
        if scale < 1:
            height = int(height * scale)
            width = int(width * scale)
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for _ in range(frames_count):
        ret, frame = cap.read()
        if not ret:
            break
        if max_side is not None:
            frame = cv2.resize(frame, (width, height))
        frames.append(frame)
    
    cap.release()
    return frames


def convert_fps(frame_indexes: List[int], base_fps: int,
                out_fps: int) -> List[int]:
    """Convert frame indexes to different FPS.
    
    Args:
        frame_indexes (List[int]): Original frame indexes
        base_fps (int): Original FPS
        out_fps (int): Target FPS
        
    Returns:
        List[int]: Converted frame indexes
    """
    if base_fps == out_fps:
        return frame_indexes
    
    fps_ratio = base_fps / out_fps
    out_indexes = []
    
    for idx in frame_indexes:
        out_idx = int(idx * fps_ratio)
        if out_idx not in out_indexes:
            out_indexes.append(out_idx)
    
    return out_indexes


def get_info_from_yolo_mark(file_path: str) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """Read YOLO mark annotation file.
    
    Args:
        file_path (str): Path to annotation file
        
    Returns:
        Tuple[List[Tuple[int, int, int, int]], List[str]]: Boxes and labels
    """
    boxes = []
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            boxes.append((x1, y1, x2, y2))
            labels.append(str(int(class_id)))
    
    return boxes, labels
