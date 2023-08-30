# -*- coding: utf-8 -*-
import os
import glob
import json
import requests
import multiprocessing as mp

import cv2
import PIL
import numpy as np
from matplotlib import pyplot as plt


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert('RGB')
    return image


def show_image(image, figsize=(5, 5), cmap=None, title='',
               xlabel=None, ylabel=None, axis=False):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis(axis)
    plt.show()


def show_images(images, n_rows=1, titles=None, figsize=(5, 5),
                cmap=None, xlabel=None, ylabel=None, axis=False):
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


def get_video_info(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return height, width, fps, frame_count


def get_img_names(folder, img_format='png'):
    img_paths = glob.glob(os.path.join(folder, f'*.{img_format}'))
    img_names = [os.path.basename(x) for x in img_paths]
    return img_names


def read_image(img_path: str, to_rgb: bool = True,
               flag: int = cv2.IMREAD_COLOR) -> np.array:
    '''
    img_path: path to image
    to_rgb: apply cv2.COLOR_BGR2RGB or not
    flag: [cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED]
    '''
    image = cv2.imread(img_path, flag)
    if image is None:
        raise FileNotFoundError(f'{img_path}')
    if to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def apply_mask_with_gauss(bg_img, src_img, mask,
                          kernel=(7, 7), sigma=0.0, alpha=0.5):
    mask = mask.astype(np.float32)
    b_mask = cv2.GaussianBlur(mask, kernel, sigma)
    b_mask = b_mask[:, :, None]
    out_image = bg_img.astype(np.float32)
    out_image = out_image * (1.0 - b_mask*alpha) + \
        src_img.astype(np.float32) * b_mask*alpha
    out_image = np.clip(out_image, 0, 255).astype(np.uint8)
    return out_image


def get_color_mask_with_hsv(image, COLOR_MIN, COLOR_MAX):
    out_img = image.copy()
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(out_img, COLOR_MIN, COLOR_MAX)
    return mask.astype(bool)


def get_mask_for_box(img_h, img_w, box):
    x1, y1, x2, y2 = box
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask.astype(bool)


def color_mask(mask, colors):
    h, w = mask.shape[:2]
    colored_image = np.ones((h, w, 3)).astype(np.uint8)*255
    for m_color in colors:
        colored_image[mask == m_color] = colors[m_color]
    return colored_image


def draw_box(image, box, label=None, color=(255, 0, 0),
             line_thickness=2, font_thickness=2, font_scale=1):
    x1, y1, x2, y2 = box

    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
    image = cv2.putText(image, label, (x1 + 10, y1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                        font_thickness, cv2.LINE_AA)

    return image


def get_boxes_intersection(box1, box2):
    dx = min(box1[2], box2[2]) - max(box1[0], box2[0])
    dy = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if (dx >= 0) and (dy >= 0):
        return dx*dy
    else:
        return 0


def change_contrast(input_img, contrast=0):
    image = input_img.copy()
    f = 131*(contrast + 127)/(127*(131-contrast))
    alpha_c = f
    gamma_c = 127*(1-f)
    image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    return image


def clear_noise(image):
    img = image.copy()

    e_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode = cv2.morphologyEx(img, cv2.MORPH_ERODE, e_kernel)

    c_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.morphologyEx(erode, cv2.MORPH_DILATE, c_kernel)

    return dilate


def resize_proportional(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    coef = 1 if img_h <= max_h and img_w <= max_w else max(
        img_h / max_h, img_w / max_w)
    h = int(img_h / coef)
    w = int(img_w / coef)
    img = cv2.resize(img, (w, h))
    return img


def make_image_padding(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    max_h = max(img_h, max_h)
    max_w = max(img_w, max_w)
    bg = np.full((max_h, max_w, img_c), 255, dtype=np.uint8)
    x1 = (max_w - img_w) // 2
    y1 = (max_h - img_h) // 2
    x2 = x1 + img_w
    y2 = y1 + img_h
    bg[y1:y2, x1:x2, :] = img.copy()
    return bg


def shift_all_colors(input_image):
    input_frame = np.array(input_image).copy()
    hsv_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2HSV)
    shift_coef = np.random.randint(0, 255)
    hsv_frame[:, :, 0] += shift_coef
    out_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
    return out_frame


def show_video_in_jupyter(video_path, width=480):
    from IPython.display import HTML
    from base64 import b64encode

    data_url = "data:video/mp4;base64," + b64encode(open(video_path, 'rb').read()).decode()
    return HTML(f'''
        <video width={width} controls>
            <source src="{data_url}" type="video/mp4">
        </video>
    ''')


def split_on_chunks(data, n_chunks):
    chunk_size = int(len(data) / n_chunks)
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    return chunks


def do_multiprocess(foo, args, n_jobs):
    with mp.Pool(n_jobs) as pool:
        out = pool.map(foo, args)
    return out


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
    return v2


def get_linear_value(current_index, start_value, total_steps, end_value=0):
    values = np.linspace(start_value, end_value, total_steps, dtype=np.float32) / start_value
    values = values * start_value
    return values[current_index]

def get_cosine_value(current_index, start_value, total_steps, end_value=0):
    values = np.linspace(end_value, total_steps, total_steps, dtype=np.float32) * np.pi / total_steps
    values = np.cos(values)
    values = (values + 1) * start_value / 2
    return values[current_index]

def get_ema_value(current_index, start_value, eta):
    value = start_value * eta ** current_index
    return value
    

def get_info_from_yolo_mark(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    info = []
    for line in lines:
        x = line.split()
        label, x, y, w, h = x[0], x[1], x[2], x[3], x[4]
        info.append({
            'label': label,
            'x1': float(x) - float(w) / 2,
            'y1': float(y) - float(h) / 2,
            'x2': float(x) + float(w) / 2,
            'y2': float(y) + float(h) / 2,
        })
    return info
