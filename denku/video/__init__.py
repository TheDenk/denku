# -*- coding: utf-8 -*-
"""Video processing utilities for the denku package."""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import os
from tqdm import tqdm


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
               max_side: Optional[int] = None,
               color_space: str = 'RGB',
               show_progress: bool = False,
               frame_stride: int = 1) -> Tuple[np.ndarray, int]:
    """Read video frames with enhanced error handling and progress tracking.

    Args:
        video_path (str): Path to video file
        start_frame (int, optional): Starting frame index. Defaults to 0.
        frames_count (Optional[int], optional): Number of frames to read. Defaults to None.
        max_side (Optional[int], optional): Maximum side length for proportional resize. Defaults to None.
        color_space (str, optional): Output color space ('BGR' or 'RGB'). Defaults to 'RGB'.
        show_progress (bool, optional): Whether to show progress bar. Defaults to False.
        frame_stride (int, optional): Read every Nth frame. Defaults to 1.

    Returns:
        Tuple[np.ndarray, int]: Numpy array of video frames [frames_count, height, width, channels] and FPS

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If start_frame or frames_count is invalid
        ValueError: If color_space is not 'BGR' or 'RGB'
        ValueError: If frame_stride is less than 1
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video file not found: {video_path}')

    if color_space not in ['BGR', 'RGB']:
        raise ValueError("color_space must be either 'BGR' or 'RGB'")

    if frame_stride < 1:
        raise ValueError('frame_stride must be at least 1')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Could not open video file: {video_path}')

    height, width, fps, total_frames = get_capture_info(cap)

    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(
            f'start_frame must be between 0 and {total_frames - 1}')

    if frames_count is None:
        frames_count = (total_frames - start_frame) // frame_stride
    elif frames_count <= 0 or start_frame + frames_count * frame_stride > total_frames:
        raise ValueError(
            f'frames_count must be between 1 and {(total_frames - start_frame) // frame_stride}')

    # Calculate new dimensions while maintaining aspect ratio
    if max_side is not None:
        scale = max_side / max(height, width)
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width = width
            new_height = height
    else:
        new_width = width
        new_height = height

    # Pre-allocate memory for frames
    frames = np.empty((frames_count, new_height, new_width, 3), dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if show_progress:
        iterator = tqdm(range(frames_count), desc='Reading video frames')
    else:
        iterator = range(frames_count)

    for i in iterator:
        ret, frame = cap.read()
        if not ret:
            # If we couldn't read a frame, truncate the array
            frames = frames[:i]
            break

        if max_side is not None and scale < 1:
            frame = cv2.resize(frame, (new_width, new_height),
                               interpolation=cv2.INTER_AREA)

        if color_space == 'RGB':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames[i] = frame

        # Skip frames according to stride
        for _ in range(frame_stride - 1):
            cap.read()

    cap.release()
    return frames, fps / frame_stride


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

    print('Opening video files...')
    for video_path in tqdm(input_videos):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f'Could not open video: {video_path}')

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
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (output_width, output_height))

    print('Processing frames...')
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
                    frame = last_frames[cell_idx] if last_frames[cell_idx] is not None else np.zeros(
                        (cell_heights[cell_idx], cell_widths[cell_idx], 3), dtype=np.uint8)
            else:
                frame = last_frames[cell_idx] if last_frames[cell_idx] is not None else np.zeros(
                    (cell_heights[cell_idx], cell_widths[cell_idx], 3), dtype=np.uint8)

            if frame.size == 0:
                frame = np.zeros(
                    (cell_heights[cell_idx], cell_widths[cell_idx], 3), dtype=np.uint8)

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
    print(f'Video grid successfully created at: {output_path}')


def overlay_video(background_path, overlay_path, output_path, overlay_scale=0.3, x_offset=10, y_offset=10):
    bg_cap = cv2.VideoCapture(background_path)
    overlay_cap = cv2.VideoCapture(overlay_path)
    
    bg_width = int(bg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    bg_height = int(bg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = bg_cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (bg_width, bg_height))
    
    while True:
        ret_bg, bg_frame = bg_cap.read()
        ret_overlay, overlay_frame = overlay_cap.read()
        
        if not ret_bg or not ret_overlay:
            break
        
        overlay_height, overlay_width = overlay_frame.shape[:2]
        new_width = int(bg_width * overlay_scale)
        new_height = int((new_width / overlay_width) * overlay_height)
        overlay_resized = cv2.resize(overlay_frame, (new_width, new_height))
        
        bg_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = overlay_resized
        out.write(bg_frame)

    bg_cap.release()
    overlay_cap.release()
    out.release()


def concatenate_videos(input_paths, output_path):
    first_video = cv2.VideoCapture(input_paths[0])
    fps = first_video.get(cv2.CAP_PROP_FPS)
    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_video.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for path in input_paths:
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame.shape[1] != width) or (frame.shape[0] != height):
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
        cap.release()

    out.release()


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
            class_id, x_center, y_center, width, height = map(
                float, line.strip().split())
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            boxes.append((x1, y1, x2, y2))
            labels.append(str(int(class_id)))

    return boxes, labels


def convert_video_fps(frames: np.ndarray, original_fps: float, target_fps: float) -> np.ndarray:
    """Convert video frames from one FPS to another by taking nearest frames.

    Args:
        frames (np.ndarray): Input video frames [frames_count, height, width, channels]
        original_fps (float): Original frame rate
        target_fps (float): Target frame rate

    Returns:
        np.ndarray: Converted video frames [new_frames_count, height, width, channels]

    Raises:
        ValueError: If original_fps or target_fps is not positive
    """
    if original_fps <= 0 or target_fps <= 0:
        raise ValueError('FPS values must be positive')

    if original_fps == target_fps:
        return frames

    fps_ratio = original_fps / target_fps

    original_frame_count = len(frames)
    target_frame_count = int(original_frame_count * target_fps / original_fps)

    out_frames = np.empty(
        (target_frame_count, *frames.shape[1:]), dtype=frames.dtype)

    for i in range(target_frame_count):
        original_pos = i * fps_ratio

        frame_idx = int(round(original_pos))
        frame_idx = min(frame_idx, original_frame_count - 1)

        out_frames[i] = frames[frame_idx]

    return out_frames
