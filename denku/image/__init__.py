"""Image processing utilities for the denku package."""

import os
import glob
from typing import Tuple, List, Optional, Union
import numpy as np
import cv2
import PIL
import requests


def download_image(url: str) -> np.ndarray:
    """Download image from URL and convert to RGB.
    
    Args:
        url (str): URL of the image
        
    Returns:
        np.ndarray: Downloaded and processed image
    """
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert('RGB')
    return np.array(image)


def read_image(img_path: str, to_rgb: bool = True,
               flag: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read image from file.
    
    Args:
        img_path (str): Path to image
        to_rgb (bool, optional): Convert BGR to RGB. Defaults to True.
        flag (int, optional): OpenCV imread flag. Defaults to cv2.IMREAD_COLOR.
        
    Returns:
        np.ndarray: Loaded image
        
    Raises:
        FileNotFoundError: If image file not found
    """
    image = cv2.imread(img_path, flag)
    if image is None:
        raise FileNotFoundError(f'{img_path}')
    if to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(img: np.ndarray, file_path: str, mkdir: bool = False) -> bool:
    """Save image to file.
    
    Args:
        img (np.ndarray): Image to save
        file_path (str): Path to save image
        mkdir (bool, optional): Create directory if not exists. Defaults to False.
        
    Returns:
        bool: True if successful
    """
    if mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img)


def get_img_names(folder: str, img_format: str = 'png') -> List[str]:
    """Get list of image filenames in folder.
    
    Args:
        folder (str): Folder path
        img_format (str, optional): Image format. Defaults to 'png'.
        
    Returns:
        List[str]: List of image filenames
    """
    img_paths = glob.glob(os.path.join(folder, f'*.{img_format}'))
    img_names = [os.path.basename(x) for x in img_paths]
    return img_names


def merge_images_by_mask_with_gauss(bg_img: np.ndarray, src_img: np.ndarray, 
                                  mask: np.ndarray, kernel: Tuple[int, int] = (7, 7),
                                  sigma: float = 0.0, alpha: float = 0.5) -> np.ndarray:
    """Merge two images using a mask with Gaussian blur.
    
    Args:
        bg_img (np.ndarray): Background image
        src_img (np.ndarray): Source image
        mask (np.ndarray): Binary mask
        kernel (Tuple[int, int], optional): Gaussian kernel size. Defaults to (7, 7).
        sigma (float, optional): Gaussian sigma. Defaults to 0.0.
        alpha (float, optional): Blend factor. Defaults to 0.5.
        
    Returns:
        np.ndarray: Merged image
    """
    mask = mask.astype(np.float32)
    b_mask = cv2.GaussianBlur(mask, kernel, sigma)
    b_mask = b_mask[:, :, None]
    out_image = bg_img.astype(np.float32)
    out_image = out_image * (1.0 - b_mask*alpha) + \
        src_img.astype(np.float32) * b_mask*alpha
    out_image = np.clip(out_image, 0, 255).astype(np.uint8)
    return out_image


def get_color_mask_with_hsv(image: np.ndarray, COLOR_MIN: np.ndarray,
                           COLOR_MAX: np.ndarray) -> np.ndarray:
    """Get binary mask from HSV color range.
    
    Args:
        image (np.ndarray): Input RGB image
        COLOR_MIN (np.ndarray): Minimum HSV values
        COLOR_MAX (np.ndarray): Maximum HSV values
        
    Returns:
        np.ndarray: Binary mask
    """
    out_img = image.copy()
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(out_img, COLOR_MIN, COLOR_MAX)
    return mask.astype(bool)


def get_mask_for_box(img_h: int, img_w: int, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Create binary mask from bounding box.
    
    Args:
        img_h (int): Image height
        img_w (int): Image width
        box (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2)
        
    Returns:
        np.ndarray: Binary mask
    """
    x1, y1, x2, y2 = box
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask.astype(bool)


def color_mask(mask: np.ndarray, colors: dict) -> np.ndarray:
    """Color binary mask with specified colors.
    
    Args:
        mask (np.ndarray): Binary mask
        colors (dict): Dictionary mapping mask values to RGB colors
        
    Returns:
        np.ndarray: Colored mask
    """
    h, w = mask.shape[:2]
    colored_image = np.ones((h, w, 3)).astype(np.uint8)*255
    for m_color in colors:
        colored_image[mask == m_color] = colors[m_color]
    return colored_image


def change_contrast(input_img: np.ndarray, contrast: float = 0) -> np.ndarray:
    """Change image contrast.
    
    Args:
        input_img (np.ndarray): Input image
        contrast (float, optional): Contrast value. Defaults to 0.
        
    Returns:
        np.ndarray: Image with adjusted contrast
    """
    image = input_img.copy()
    f = 131*(contrast + 127)/(127*(131-contrast))
    alpha_c = f
    gamma_c = 127*(1-f)
    image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    return image


def clear_noise(image: np.ndarray) -> np.ndarray:
    """Remove noise from binary image using morphological operations.
    
    Args:
        image (np.ndarray): Binary input image
        
    Returns:
        np.ndarray: Cleaned image
    """
    img = image.copy()

    e_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode = cv2.morphologyEx(img, cv2.MORPH_ERODE, e_kernel)

    c_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.morphologyEx(erode, cv2.MORPH_DILATE, c_kernel)

    return dilate


def apply_divide_factor(x: int, divide_factor: int = 8, upward: bool = True) -> int:
    """Apply division factor to number.
    
    Args:
        x (int): Input number
        divide_factor (int, optional): Division factor. Defaults to 8.
        upward (bool, optional): Round up if True, down if False. Defaults to True.
        
    Returns:
        int: Processed number
    """
    if upward:
        return (x // divide_factor + int(x % divide_factor != 0)) * divide_factor 
    return x // divide_factor * divide_factor


def resize_to_min_sides(input_image: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    """Resize image to have minimum sides.
    
    Args:
        input_image (np.ndarray): Input image
        min_h (int): Minimum height
        min_w (int): Minimum width
        
    Returns:
        np.ndarray: Resized image
    """
    image = np.array(input_image)
    img_h, img_w = image.shape[:2]
    
    if img_h >= min_h and img_w >= min_w:
        coef = min(min_h / img_h, min_w / img_w)
    elif img_h <= min_h and img_w <= min_w:
        coef = max(min_h / img_h, min_w / img_w)
    else:
        coef = min_h / img_h if min_h > img_h else min_w / img_w 

    out_h, out_w = int(img_h * coef), int(img_w * coef)
    image = cv2.resize(image, (out_w, out_h))
    return image


def resize_to_min_side(input_image: np.ndarray, min_side: int,
                       interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Resize image to have minimum side.
    
    Args:
        input_image (np.ndarray): Input image
        min_side (int): Minimum side
        interpolation (int, optional): Interpolation method. Defaults to cv2.INTER_CUBIC.

    Returns:
        np.ndarray: Resized image
    """
    image = input_image.copy()
    h, w = image.shape[:2]
    cur_side = min(h, w)
    coef = min_side / cur_side
    new_h, new_w = [int(x * coef) for x in [h, w]]
    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return image


def resize_to_max_side(input_image: np.ndarray, max_side: int,
                       interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Resize image to have maximum side.
    
    Args:
        input_image (np.ndarray): Input image
        max_side (int): Maximum side
        interpolation (int, optional): Interpolation method. Defaults to cv2.INTER_CUBIC.
        
    Returns:
        np.ndarray: Resized image
    """
    image = input_image.copy()
    h, w = image.shape[:2]
    cur_side = max(h, w)
    coef = max_side / cur_side
    new_h, new_w = [int(x * coef) for x in [h, w]]
    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return image