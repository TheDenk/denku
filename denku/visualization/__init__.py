"""Visualization utilities for the denku package."""

from typing import List, Optional, Tuple, Union
import numpy as np
import cv2
import PIL
from matplotlib import pyplot as plt
from IPython.display import HTML
from base64 import b64encode


def show_image(image: np.ndarray, figsize: Tuple[int, int] = (5, 5),
               cmap: Optional[str] = None, title: str = '',
               xlabel: Optional[str] = None, ylabel: Optional[str] = None,
               axis: bool = False) -> None:
    """Display single image using matplotlib.
    
    Args:
        image (np.ndarray): Image to display
        figsize (Tuple[int, int], optional): Figure size. Defaults to (5, 5).
        cmap (Optional[str], optional): Colormap. Defaults to None.
        title (str, optional): Plot title. Defaults to ''.
        xlabel (Optional[str], optional): X-axis label. Defaults to None.
        ylabel (Optional[str], optional): Y-axis label. Defaults to None.
        axis (bool, optional): Show axis. Defaults to False.
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis(axis)
    plt.show()


def show_images(images: List[np.ndarray], n_rows: int = 1,
                titles: Optional[Union[str, List[str]]] = None,
                figsize: Tuple[int, int] = (5, 5),
                cmap: Optional[str] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                axis: bool = False) -> None:
    """Display multiple images in a grid using matplotlib.
    
    Args:
        images (List[np.ndarray]): List of images to display
        n_rows (int, optional): Number of rows in grid. Defaults to 1.
        titles (Optional[Union[str, List[str]]], optional): Plot titles. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (5, 5).
        cmap (Optional[str], optional): Colormap. Defaults to None.
        xlabel (Optional[str], optional): X-axis label. Defaults to None.
        ylabel (Optional[str], optional): Y-axis label. Defaults to None.
        axis (bool, optional): Show axis. Defaults to False.
    """
    n_cols = len(images) // n_rows
    if n_rows == n_cols == 1:
        if isinstance(titles, str) or titles is None:
            title = titles
        if isinstance(titles, list):
            title = titles[0]
        show_image(images[0], title=title, figsize=figsize,
                   cmap=cmap, xlabel=xlabel, ylabel=ylabel, axis=axis)
    else:
        titles = titles if isinstance(titles, list) else [
            '' for _ in range(len(images))]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.tight_layout(pad=0.0)
        axes = axes.flatten()
        for index, ax in enumerate(axes):
            ax.imshow(images[index], cmap=cmap)
            ax.set_title(titles[index])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.axis(axis)
        plt.show()


def show_video_in_jupyter(video_path: str, width: int = 480) -> HTML:
    """Display video in Jupyter notebook.
    
    Args:
        video_path (str): Path to video file
        width (int, optional): Video width. Defaults to 480.
        
    Returns:
        HTML: HTML element with video
    """
    data_url = "data:video/mp4;base64," + b64encode(open(video_path, 'rb').read()).decode()
    return HTML(f'''
        <video width={width} controls>
            <source src="{data_url}" type="video/mp4">
        </video>
    ''')


def show_gif_in_jupyter(gif_path: str, width: int = 480) -> HTML:
    """Display GIF in Jupyter notebook.
    
    Args:
        gif_path (str): Path to GIF file
        width (int, optional): GIF width. Defaults to 480.
        
    Returns:
        HTML: HTML element with GIF
    """
    return HTML(f'<img src="{gif_path}" width="{width}">')


def draw_image_title(input_image: np.ndarray, text: str,
                    color: Optional[List[int]] = None,
                    font_thickness: int = 2) -> np.ndarray:
    """Add title to numpy image.
    
    Args:
        input_image np.ndarray: Input image
        text (str): Title text
        color (Optional[List[int]], optional): Text color. Defaults to None.
        font_thickness (int, optional): Font thickness. Defaults to 2.
        
    Returns:
        np.ndarray: Image with title
    """
    out_image = input_image.copy()
    img_h, img_w = out_image.shape[:2]
    
    font_scale = max(font_thickness // 2, 1)
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    text_w, text_h = cv2.getTextSize(text, 0, fontScale=font_scale, thickness=font_thickness)[0]
    
    pad = text_h + text_h // 2
    text_x, text_y = (img_w - text_w) // 2, text_h + text_h // 4
    
    out_image[:pad, :] = np.clip((out_image[:pad, :].astype(np.uint16) + 64), 0, 255).astype(np.uint8)
    out_image = cv2.putText(out_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)
    return out_image


def draw_box(input_image: np.ndarray, box: Tuple[int, int, int, int],
             label: Optional[str] = None, color: Tuple[int, int, int] = (255, 0, 0),
             line_thickness: Optional[int] = 3,
             font_thickness: Optional[int] = None,
             font_scale: Optional[float] = None) -> np.ndarray:
    """Draw bounding box on image.
    
    Args:
        input_image (np.ndarray): Input image
        box (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2)
        label (Optional[str], optional): Box label. Defaults to None.
        color (Tuple[int, int, int], optional): Box color. Defaults to (255, 0, 0).
        line_thickness (Optional[int], optional): Line thickness. Defaults to 3.
        font_thickness (Optional[int], optional): Font thickness. Defaults to None.
        font_scale (Optional[float], optional): Font scale. Defaults to None.
        
    Returns:
        np.ndarray: Image with drawn box
    """
    x1, y1, x2, y2 = box
    image = input_image.copy()

    line_thickness = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

    if label:
        font_scale = font_scale or line_thickness / 3
        font_thickness = font_thickness or max(line_thickness - 1, 1) 
        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        y1 = t_size[1] + 3 if y1 < (t_size[1] + 3) else y1
        t_x2, t_y2 = x1 + t_size[0], y1 - t_size[1] - 3
        image = cv2.rectangle(image, (x1, y1), (t_x2, t_y2), color, -1, cv2.LINE_AA)  # filled
        
        image = cv2.putText(image, label, (x1, y1 - 2), 0, font_scale, [0, 0, 0], thickness=font_thickness, lineType=cv2.LINE_AA)
        image = cv2.putText(image, label, (x1, y1 - 2), 0, font_scale, [225, 255, 255], thickness=font_thickness - 1, lineType=cv2.LINE_AA)
    return image


def add_mask_on_image(image: np.ndarray, mask: np.ndarray,
                     color: List[int], alpha: float = 0.9) -> np.ndarray:
    """Add colored mask overlay on image.
    
    Args:
        image (np.ndarray): Input image
        mask (np.ndarray): Binary mask
        color (List[int]): Mask color
        alpha (float, optional): Blend factor. Defaults to 0.9.
        
    Returns:
        np.ndarray: Image with mask overlay
    """
    color = np.array(color)
    original_mask = mask.copy()
    if original_mask.ndim == 2:
        original_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        original_mask = np.moveaxis(original_mask, 0, -1)
    elif original_mask.shape[-1] == 1:
        original_mask = np.concatenate([original_mask] * 3, axis=2)
        
    colored_mask = original_mask.astype(np.float32) / 255 * color
    colored_mask = np.clip(colored_mask, 0, 255).astype(np.uint8)
    image_combined = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    return image_combined
