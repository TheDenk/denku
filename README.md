## DENK UTILS

Custom computer vision utilities for image and video processing, visualization, and memory management.

[![PyPI Downloads](https://static.pepy.tech/badge/denku)](https://pepy.tech/projects/denku)
## Installation

### For Users
```bash
pip install denku
```

## Features

### Image Processing
```python
from denku import read_image, save_image, change_contrast

# Read and process images
image = read_image("path/to/image.jpg")
processed_image = change_contrast(image, contrast=20)
save_image(processed_image, "path/to/output.jpg")

# Create and apply masks
from denku import get_color_mask_with_hsv, merge_images_by_mask_with_gauss
import numpy as np

# Create a mask based on HSV color range
color_min = np.array([0, 100, 100])
color_max = np.array([10, 255, 255])
mask = get_color_mask_with_hsv(image, color_min, color_max)

# Merge images using a mask
background = read_image("path/to/background.jpg")
merged = merge_images_by_mask_with_gauss(background, image, mask)

# Resize images
from denku import resize_to_min_side, resize_to_max_side

# Proportionally resize image to have minimum side of 256 pixels
resized_min = resize_to_min_side(image, min_side=256)

# Proportionally resize image to have maximum side of 512 pixels
resized_max = resize_to_max_side(image, max_side=512)

# Center crop and rotate images
from denku import center_crop, rotate_image, show_images

# Get original dimensions
h, w = image.shape[:2]

# Crop center 50% of the image
target_h = h // 2
target_w = w // 2
cropped = center_crop(image, target_h, target_w)

# Rotate image by different angles
rotated_90 = rotate_image(image, angle=90)    # Rotate 90 degrees counter-clockwise
rotated_45 = rotate_image(image, angle=45)    # Rotate 45 degrees counter-clockwise
rotated_neg30 = rotate_image(image, angle=-30) # Rotate 30 degrees clockwise

# Display all images in a grid
show_images(
    images=[image, cropped, rotated_90, rotated_45, rotated_neg30],
    n_rows=2,
    titles=['Original', 'Center Cropped', '90°', '45°', '-30°'],
    global_title='Image Processing Examples'
)
```

### Video Processing
```python
import denku

# Read video frames
# Read every frame
frames, fps = denku.read_video("video.mp4")

# Read every 2nd frame
frames, fps = denku.read_video("video.mp4", frame_stride=2)

# Read every 5th frame
frames, fps = denku.read_video("video.mp4", frame_stride=5)

# Convert video to different FPS
converted_frames = denku.convert_video_fps(frames, original_fps=fps, target_fps=16)
denku.save_video(converted_frames, "output.mp4", fps=16)

# Create a grid of videos
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
grid = denku.create_video_grid(video_paths, grid_size=(2, 2))
denku.save_video(grid, "video_grid.mp4")

# Advanced video operations
from denku import overlay_video, concatenate_videos, save_video_from_frames

# Overlay one video on top of another
overlay_video(
    background_path='path/to/video1.mp4',
    overlay_path='path/to/video2.mp4',
    output_path='output/overlayed.mp4',
    overlay_scale=0.3,  # Make overlay 30% of original size
    x_offset=10,        # 10 pixels from left
    y_offset=10         # 10 pixels from top
)

# Concatenate videos horizontally
concatenate_videos(
    input_paths=['path/to/video1.mp4', 'path/to/video2.mp4'],
    output_path='output/concatenated.mp4'
)

# Save frames as video
frames, fps = read_video('path/to/video.mp4')
save_video_from_frames(
    frames=frames,
    output_path='output/saved.mp4',
    fps=fps,
    codec='mp4v',
    input_format='rgb'
)
```

### Box Operations
```python
from denku import get_boxes_iou, get_boxes_intersection, get_boxes_union, get_boxes_iou_matrix

# Example boxes (x1, y1, x2, y2)
box1 = (10, 10, 50, 50)  # First box
box2 = (30, 30, 70, 70)  # Second box

# Calculate IoU (Intersection over Union)
iou = get_boxes_iou(box1, box2)
print(f"IoU: {iou:.2f}")  # Should be around 0.25

# Calculate intersection area
intersection = get_boxes_intersection(box1, box2)
print(f"Intersection area: {intersection}")  # Should be 400

# Calculate union area
union = get_boxes_union(box1, box2)
print(f"Union area: {union}")  # Should be 1600

# Calculate IoU matrix between two sets of boxes
boxes1 = [
    (10, 10, 50, 50),   # First box
    (20, 20, 60, 60)    # Second box
]
boxes2 = [
    (30, 30, 70, 70),   # Third box
    (40, 40, 80, 80)    # Fourth box
]

# Calculate IoU matrix between two sets of boxes
iou_matrix = get_boxes_iou_matrix(boxes1, boxes2)
print("IoU Matrix:")
print(iou_matrix)
# Output will be a 2x2 matrix showing IoU between each pair of boxes
```

### Visualization in Jupyter Notebooks
```python
from denku import show_image, show_images, show_video_in_jupyter, show_gif_in_jupyter

# Display images
show_image(image, title="My Image", figsize=(10, 10))

# Display multiple images in a grid
show_images([image1, image2, image3], n_rows=2, titles=["Image 1", "Image 2", "Image 3"])

# Display video in Jupyter
show_video_in_jupyter("path/to/video.mp4", width=640)

# Display GIF in Jupyter
show_gif_in_jupyter("path/to/animation.gif", width=480)
```

### Memory Management
```python
from denku import empty_cuda_cache, print_cuda_allocated_memory

# Reset CUDA memory and run garbage collection
empty_cuda_cache()

# Print current CUDA memory usage
print_cuda_allocated_memory()
```


## Development

### Setup Development Environment
```bash
# Clone the repository
git clone https://github.com/TheDenk/denku.git
cd denku

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=denku

# Run tests for a specific module
pytest tests/test_utils.py

# Run tests with detailed output
pytest -v

# Run tests and show coverage in HTML format
pytest --cov=denku --cov-report=html
```


## Contacts
<p>Issues should be raised directly in the repository. For professional support and recommendations please <a>welcomedenk@gmail.com</a>.</p>
