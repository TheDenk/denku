"""General utility functions for the denku package."""

import datetime
import json
from typing import Any, List, Union
import numpy as np


def get_datetime() -> str:
    """Get current datetime in UTC+3 timezone.
    
    Returns:
        str: Formatted datetime string in format 'YYYY-MM-DD_HH-MM-SS'
    """
    UTC = datetime.timezone(datetime.timedelta(hours=+3))
    date = datetime.datetime.now(UTC).strftime('%Y-%m-%d_%H-%M-%S')
    return date


def load_json(file_path: str) -> Any:
    """Load JSON file.
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        Any: Parsed JSON content
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def split_on_chunks(data: List[Any], n_chunks: int) -> List[List[Any]]:
    """Split data into n chunks.
    
    Args:
        data (List[Any]): List to split
        n_chunks (int): Number of chunks to split into
        
    Returns:
        List[List[Any]]: List of chunks
    """
    return np.array_split(data, n_chunks)


def get_linear_value(current_index: int, start_value: float, 
                    total_steps: int, end_value: float = 0) -> float:
    """Calculate linear interpolation value.
    
    Args:
        current_index (int): Current step index
        start_value (float): Starting value
        total_steps (int): Total number of steps
        end_value (float, optional): Ending value. Defaults to 0.
        
    Returns:
        float: Interpolated value
    """
    return start_value + (end_value - start_value) * current_index / total_steps


def get_cosine_value(current_index: int, start_value: float,
                    total_steps: int, end_value: float = 0) -> float:
    """Calculate cosine interpolation value.
    
    Args:
        current_index (int): Current step index
        start_value (float): Starting value
        total_steps (int): Total number of steps
        end_value (float, optional): Ending value. Defaults to 0.
        
    Returns:
        float: Interpolated value
    """
    progress = current_index / total_steps
    return end_value + (start_value - end_value) * 0.5 * (1 + np.cos(np.pi * progress))


def get_ema_value(current_index: int, start_value: float, eta: float) -> float:
    """Calculate exponential moving average value.
    
    Args:
        current_index (int): Current step index
        start_value (float): Starting value
        eta (float): Smoothing factor
        
    Returns:
        float: EMA value
    """
    return start_value * (1 - eta) ** current_index


def get_boxes_intersection(box1, box2):
    """Calculate intersection area of two boxes.
    
    Args:
        box1 (Tuple[int, int, int, int]): First box
        box2 (Tuple[int, int, int, int]): Second box
        
    Returns:
        float: Intersection area
    """
    dx = min(box1[2], box2[2]) - max(box1[0], box2[0])
    dy = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if (dx >= 0) and (dy >= 0):
        return dx*dy
    else:
        return 0


def get_boxes_union(box1, box2):
    """Calculate union area of two boxes.
    
    Args:
        box1 (Tuple[int, int, int, int]): First box
        box2 (Tuple[int, int, int, int]): Second box
        
    Returns:
        float: Union area
    """
    return (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - get_boxes_intersection(box1, box2)

def get_boxes_iou(box1, box2):
    """Calculate IoU between two boxes.
    
    Args:
        box1 (Tuple[int, int, int, int]): First box
        box2 (Tuple[int, int, int, int]): Second box    
        
    Returns:
        float: IoU value    
    """
    return get_boxes_intersection(box1, box2) / get_boxes_union(box1, box2)


def get_boxes_iou_matrix(boxes1, boxes2):
    """Calculate IoU matrix between two sets of boxes.
    
    Args:
        boxes1 (List[Tuple[int, int, int, int]]): List of boxes
        boxes2 (List[Tuple[int, int, int, int]]): List of boxes
        
    Returns:
        np.ndarray: IoU matrix
    """
    iou_matrix = np.zeros((len(boxes1), len(boxes2)))
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou_matrix[i, j] = get_boxes_iou(box1, box2)
    return iou_matrix

