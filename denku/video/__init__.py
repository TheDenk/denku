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


def create_video_grid(
    input_videos: List[str],
    output_path: str,
    grid_size: Tuple[int, int] = (2, 2),
    target_duration: float = 5.0,
    fps: int = 30,
    target_cell_width: Optional[int] = None,
    target_cell_height: Optional[int] = None,
    target_output_width: Optional[int] = None,
    target_output_height: Optional[int] = None,
    padding_color: Tuple[int, int, int] = (0, 0, 0)
):
    """
    Create an NxN video grid.
    
    Args:
        input_videos: List of input video paths
        output_path: Output video file path
        grid_size: Tuple of (rows, cols) for the grid
        target_duration: Duration in seconds for each video
        fps: Output video frame rate
        target_cell_width: Width for each grid cell
        target_cell_height: Height for each grid cell
        target_output_width: Total output width
        target_output_height: Total output height
        padding_color: Background color for padding (BGR format)
    """
    rows, cols = grid_size
    total_cells = rows * cols
    
    if len(input_videos) < total_cells:
        last_video = input_videos[-1]
        input_videos += [last_video] * (total_cells - len(input_videos))
    elif len(input_videos) > total_cells:
        input_videos = input_videos[:total_cells]
    
    caps = []
    frame_counts = []
    original_fps = []
    cell_widths = []
    cell_heights = []
    
    print("Opening video files...")
    for video_path in tqdm(input_videos):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        caps.append(cap)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counts.append(frame_count)
        original_fps.append(cap.get(cv2.CAP_PROP_FPS))
        cell_widths.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cell_heights.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    target_frame_count = int(target_duration * fps)
    
    if target_cell_width is None:
        cell_width = max(cell_widths)
    else:
        cell_width = target_cell_width
    
    if target_cell_height is None:
        cell_height = max(cell_heights)
    else:
        cell_height = target_cell_height
    
    if target_output_width is None:
        output_width = cell_width * cols
    else:
        output_width = target_output_width
        cell_width = output_width // cols
    
    if target_output_height is None:
        output_height = cell_height * rows
    else:
        output_height = target_output_height
        cell_height = output_height // rows
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    print("Processing frames...")
    last_frames = [None] * total_cells

    for frame_idx in tqdm(range(target_frame_count)):
        current_time = frame_idx / fps
        grid_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        grid_frame[:] = padding_color
        
        for cell_idx in range(total_cells):
            cap = caps[cell_idx]
            i = cell_idx // cols  # row index
            j = cell_idx % cols   # column index
            
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width
            
            target_frame_pos = int(current_time * original_fps[cell_idx])
    
            if target_frame_pos < frame_counts[cell_idx]:                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_pos)
                ret, frame = cap.read()
                if ret:
                    last_frames[cell_idx] = frame
                else:
                    frame = last_frames[cell_idx] if last_frames[cell_idx] is not None else np.zeros((cell_heights[cell_idx], cell_widths[cell_idx], 3), dtype=np.uint8)
            else:
                frame = last_frames[cell_idx] if last_frames[cell_idx] is not None else np.zeros((cell_heights[cell_idx], cell_widths[cell_idx], 3), dtype=np.uint8)
            
            if frame.size == 0:
                frame = np.zeros((cell_heights[cell_idx], cell_widths[cell_idx], 3), dtype=np.uint8)
 
            h, w = frame.shape[:2]
            aspect_ratio = w / h
            
            if aspect_ratio > (cell_width / cell_height):
                new_w = cell_width
                new_h = int(new_w / aspect_ratio)
            else:
                new_h = cell_height
                new_w = int(new_h * aspect_ratio)
            
            resized_frame = cv2.resize(frame, (new_w, new_h))
            
            
            pad_top = (cell_height - new_h) // 2
            pad_bottom = cell_height - new_h - pad_top
            pad_left = (cell_width - new_w) // 2
            pad_right = cell_width - new_w - pad_left
            
            cell_frame = cv2.copyMakeBorder(
                resized_frame,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=padding_color
            )
            
            grid_frame[y_start:y_end, x_start:x_end] = cell_frame
        
        out.write(grid_frame)
    
    for cap in caps:
        cap.release()
    out.release()
    print(f"Video grid successfully created at: {output_path}")
    

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
