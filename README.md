## DENK UTILS

Custom computer vision utilities for image and video processing, visualization, and memory management.


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
```

### Video Processing
```python
import denku

# Read video frames
frames, fps = denku.read_video("video.mp4")

# Convert video to different FPS
converted_frames = denku.convert_video_fps(frames, original_fps=fps, target_fps=16)
denku.save_video(converted_frames, "output.mp4", fps=16)

# Create a grid of videos
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
grid = denku.create_video_grid(video_paths, grid_size=(2, 2))
denku.save_video(grid, "video_grid.mp4")
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
