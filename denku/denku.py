# -*- coding: utf-8 -*-
import os
import gc
import glob
import json
import datetime
import requests
import multiprocessing as mp

import cv2
import PIL
import numpy as np
from matplotlib import pyplot as plt


def get_datetime():
    UTC = datetime.timezone(datetime.timedelta(hours=+3))
    date = datetime.datetime.now(UTC).strftime('%Y-%m-%d_%H-%M-%S')
    return date


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


def show_video_in_jupyter(video_path, width=480):
    from IPython.display import HTML
    from base64 import b64encode

    data_url = "data:video/mp4;base64," + b64encode(open(video_path, 'rb').read()).decode()
    return HTML(f'''
        <video width={width} controls>
            <source src="{data_url}" type="video/mp4">
        </video>
    ''')


def show_gif_in_jupyter(gif_path, width=480):
    from IPython.display import HTML
    return HTML(f'<img src="{gif_path}" width="{width}">')


def get_capture_info(cap):
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


def save_image(img, file_path, mkdir=True):
    if mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img)


def merge_images_by_mask_with_gauss(bg_img, src_img, mask,
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


def draw_box(input_image, box, label=None, color=(255, 0, 0),
             line_thickness=3, font_thickness=None, font_scale=None):
    x1, y1, x2, y2 = box
    image = input_image.copy()

    line_thickness = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

    if label:
        font_thickness = max(line_thickness - 1, 1) 
        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        t_x2, t_y2 = x1 + t_size[0], y1 - t_size[1] - 3
        image = cv2.rectangle(image, (x1, y1), (t_x2, t_y2), color, -1, cv2.LINE_AA)  # filled
        image = cv2.putText(image, label, (x1, y1 - 2), 0, line_thickness / 3, [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)
    return image


def add_mask_on_image(image, mask, color, alpha=0.9):
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


def apply_divide_factor(x, divide_factor=8, upward=True):
    if upward:
        return (x // divide_factor + int(x % divide_factor != 0)) * divide_factor 
    return x // divide_factor * divide_factor 


def resize_to_min_sides(input_image, min_h, min_w):
    image = np.array(input_image)
    img_h, img_w = image.shape[:2]
    
    if img_h >= min_h and img_w >= min_w:
        coef = min(min_h / img_h, min_w / img_w)
    elif img_h <= min_h and img_w <=min_w:
        coef = max(min_h / img_h, min_w / img_w)
    else:
        coef = min_h / img_h if min_h > img_h else min_w / img_w 

    out_h, out_w = int(img_h * coef), int(img_w * coef)
    image = cv2.resize(image, (out_w, out_h))
    return Image.fromarray(image)
    

def resize_to_min_side(input_image, min_side, interpolation=cv2.INTER_CUBIC):
    image = np.array(input_image).copy()
    h, w = image.shape[:2]
    cur_side = min(h, w)
    coef = min_side / cur_side
    new_h, new_w = [int(x * coef) for x in [h, w]]
    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return Image.fromarray(image)


def resize_to_max_side(input_image, max_side, interpolation=cv2.INTER_CUBIC):
    image = np.array(input_image).copy()
    h, w = image.shape[:2]
    cur_side = max(h, w)
    coef = max_side / cur_side
    new_h, new_w = [int(x * coef) for x in [h, w]]
    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return Image.fromarray(image)


def center_crop(image, crop_h, crop_w):
    img = np.array(image).copy()
    h, w = img.shape[:2]
    center_h = h // 2
    center_w = w // 2
    half_crop_h = crop_h // 2
    half_crop_w = crop_w // 2

    y_min = center_h - half_crop_h
    y_max = center_h + half_crop_h + crop_h % 2
    x_min = center_w - half_crop_w
    x_max = center_w + half_crop_w + crop_w % 2
    img = img[y_min:y_max, x_min:x_max]
    return PIL.Image.fromarray(img)


def resize_if_larger(image, max_h, max_w):
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
    

def split_on_chunks(data, n_chunks):
    chunk_size = len(data) // n_chunks + 1
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


def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    return img.reshape(shape).T
    
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


def get_module_parameters_count_m(module):
    params = [p.numel() for n, p in module.named_parameters()]
    return sum(params) / 1e6


def get_current_cuda_allocated_memory_gb():
    import torch
    return torch.cuda.memory_allocated() / 1e9


def get_module_memory_gb(module, dtype='fp32'):
    params = [p.numel() for n, p in module.named_parameters()]
    if dtype == 'fp16':
        return sum(params) * 2 / 1e9
    elif dtype == 'fp32':
        return sum(params) * 4 / 1e9
    else:
        raise NotImplementedError


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()


def print_memory():
    memory = round(torch.cuda.memory_allocated() / 1024**3, 2)
    max_memory = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    max_reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    print(f"{memory=} GB")
    print(f"{max_memory=} GB")
    print(f"{max_reserved=} GB")
