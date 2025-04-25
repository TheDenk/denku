# -*- coding: utf-8 -*-
"""Denku - A collection of computer vision utilities."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from denku.utils import (
        get_datetime,
        load_json,
        split_on_chunks,
        get_linear_value,
        get_cosine_value,
        get_ema_value
    )
    from denku.image import (
        download_image,
        read_image,
        save_image,
        get_img_names,
        merge_images_by_mask_with_gauss,
        get_color_mask_with_hsv,
        get_mask_for_box,
        color_mask,
        change_contrast,
        clear_noise,
        apply_divide_factor,
        resize_to_min_sides,
        resize_to_min_side,
        resize_to_max_side
    )
    from denku.visualization import (
        show_image,
        show_images,
        show_video_in_jupyter,
        show_gif_in_jupyter,
        draw_image_title,
        draw_box,
        add_mask_on_image
    )
    from denku.video import (
        get_capture_info,
        read_video,
        convert_fps,
        get_info_from_yolo_mark,
        create_video_grid,
        convert_video_fps
    )
    from denku.memory import (
        get_module_parameters_count_m,
        get_current_cuda_allocated_memory_gb,
        get_module_memory_gb,
        log_trainable_params,
        print_trainable_parameters,
        empty_cuda_cache,
        print_cuda_allocated_memory
    )

__version__ = '0.1.2'


def __getattr__(name: str) -> object:
    """Lazy import of submodules."""
    if name in {
        'get_datetime', 'load_json', 'split_on_chunks', 'get_linear_value',
        'get_cosine_value', 'get_ema_value'
    }:
        from denku.utils import __dict__ as utils_dict
        return utils_dict[name]

    if name in {
        'download_image', 'read_image', 'save_image', 'get_img_names',
        'merge_images_by_mask_with_gauss', 'get_color_mask_with_hsv',
        'get_mask_for_box', 'color_mask', 'change_contrast', 'clear_noise',
        'apply_divide_factor', 'resize_to_min_sides', 'resize_to_min_side',
        'resize_to_max_side'
    }:
        from denku.image import __dict__ as image_dict
        return image_dict[name]

    if name in {
        'show_image', 'show_images', 'show_video_in_jupyter',
        'show_gif_in_jupyter', 'draw_image_title', 'draw_box',
        'add_mask_on_image'
    }:
        from denku.visualization import __dict__ as vis_dict
        return vis_dict[name]

    if name in {
        'get_capture_info', 'read_video', 'convert_fps',
        'get_info_from_yolo_mark', 'create_video_grid', 'convert_video_fps'
    }:
        from denku.video import __dict__ as video_dict
        return video_dict[name]

    if name in {
        'get_module_parameters_count_m', 'get_current_cuda_allocated_memory_gb',
        'get_module_memory_gb', 'log_trainable_params',
        'print_trainable_parameters', 'empty_cuda_cache',
        'print_cuda_allocated_memory'
    }:
        from denku.memory import __dict__ as memory_dict
        return memory_dict[name]

    raise AttributeError(f"module 'denku' has no attribute '{name}'")
