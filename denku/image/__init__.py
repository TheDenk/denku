# -*- coding: utf-8 -*-
"""Image processing utilities for the denku package."""

import os
import glob
from typing import Tuple, List, Optional, Union, Dict
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
        image (np.ndarray): Input RGB image of shape (H, W, 3) with values in range [0, 255]
        COLOR_MIN (np.ndarray): Minimum HSV values of shape (3,) with values in ranges:
            H: [0, 179], S: [0, 255], V: [0, 255]
        COLOR_MAX (np.ndarray): Maximum HSV values of shape (3,) with values in ranges:
            H: [0, 179], S: [0, 255], V: [0, 255]

    Returns:
        np.ndarray: Binary mask of shape (H, W) with boolean values

    Note:
        The input image is converted from RGB to HSV color space before thresholding.
        The output mask is True where the HSV values fall within the specified range.
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


def color_mask(mask: np.ndarray, colors: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    """Color binary mask with specified colors.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W) with integer values
        colors (Dict[int, Tuple[int, int, int]]): Dictionary mapping mask values to RGB colors
            where keys are mask values and values are RGB tuples (R, G, B)

    Returns:
        np.ndarray: Colored mask of shape (H, W, 3) with RGB values
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
        image (np.ndarray): Binary input image of shape (H, W) with values 0 or 255

    Returns:
        np.ndarray: Cleaned binary image of shape (H, W) with values 0 or 255

    Note:
        This function applies erosion followed by dilation (opening operation)
        to remove small noise artifacts from the binary image.
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


def center_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Crop the center of a numpy array image to target dimensions.

    Args:
        img: Input image as numpy array (H, W, C) or (H, W)
        target_h: Target height (must be <= image height)
        target_w: Target width (must be <= image width)

    Returns:
        Cropped center image as numpy array
    """
    h, w = img.shape[:2]

    if target_h > h or target_w > w:
        raise ValueError(
            f'Target dimensions ({target_h}, {target_w}) must be smaller than '
            f'image dimensions ({h}, {w})'
        )

    start_y = h // 2 - target_h // 2
    start_x = w // 2 - target_w // 2

    return img[start_y:start_y + target_h, start_x:start_x + target_w]


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by a specified angle around its center.

    Args:
        image (np.ndarray): Input image of shape (H, W, C) or (H, W) with values in range [0, 255]
        angle (float): Rotation angle in degrees. Positive values rotate counter-clockwise.

    Returns:
        np.ndarray: Rotated image with the same shape and data type as the input image

    Note:
        The rotation is performed around the center of the image.
        The output image maintains the same dimensions as the input image,
        with black padding added where necessary.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def set_random_pixel_values(image: np.ndarray, alpha: float, value:int = 0) -> np.ndarray:
    """
    Apply salt augmentation to a numpy image by adding white pixels randomly.
    
    Args:
        image: Input image as a numpy array (H, W) or (H, W, C).
        alpha: Percentage of pixels to replace with value (0.0 to 1.0).
        valeu: Pixel value (For example: 255 - salt augmentation, 0 - paper augmentation)
    
    Returns:
        Augmented image as numpy array with same shape as input.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1")

    changed_image = image.copy()
    total_pixels = changed_image.size if changed_image.ndim == 2 else changed_image.shape[0] * changed_image.shape[1]
    num_changed_pixels = int(alpha * total_pixels)
    
    if changed_image.ndim == 2: 
        coords = [np.random.randint(0, i, num_changed_pixels) for i in changed_image.shape]
        changed_image[coords[0], coords[1]] = value
    else: 
        coords = [np.random.randint(0, i, num_changed_pixels) for i in changed_image.shape[:2]]
        changed_image[coords[0], coords[1]] = [value] * changed_image.shape[2]
    return changed_image


def paste_random_on_black_bg(
    image: np.ndarray,
    scale_range: tuple = (0.8, 1.0),
) -> np.ndarray:
    """
    Randomly scales an image and pastes it at a random position on a black background.
    
    Args:
        image: Input image as numpy array
        scale_range: Tuple (min_scale, max_scale) for random scaling range (0.8 = 80% of original size)
    
    Returns:
        Numpy array with the scaled image placed randomly on black background
    """
    background = np.zeros_like(image)
    
    scale = np.random.uniform(*scale_range)
    orig_h, orig_w = image.shape[:2]
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    y_max = max(0, orig_h - new_h)
    x_max = max(0, orig_w - new_w)
    
    pos_y = np.random.randint(0, y_max) if y_max > 0 else 0
    pos_x = np.random.randint(0, x_max) if x_max > 0 else 0

    background[pos_y:pos_y+new_h, pos_x:pos_x+new_w] = resized_img
    return background
