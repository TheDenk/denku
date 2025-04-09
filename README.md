## DENK UTILS

Custom computer vision utilities for image and video processing, visualization, and memory management.


## Getting Started
```
pip install denku
```


## Features

### Image Processing
```python
from denku.image import read_image, save_image, change_contrast

# Read and process images
image = read_image("path/to/image.jpg")
processed_image = change_contrast(image, contrast=20)
save_image(processed_image, "path/to/output.jpg")

# Create and apply masks
from denku.image import get_color_mask_with_hsv, merge_images_by_mask_with_gauss
import numpy as np

# Create a mask based on HSV color range
color_min = np.array([0, 100, 100])
color_max = np.array([10, 255, 255])
mask = get_color_mask_with_hsv(image, color_min, color_max)

# Merge images using a mask
background = read_image("path/to/background.jpg")
merged = merge_images_by_mask_with_gauss(background, image, mask)
```

### Video Processing
```python
from denku.video import read_video, convert_fps

# Read video frames
frames = read_video("path/to/video.mp4", start_frame=0, frames_count=100)

# Convert frame indexes to different FPS
original_fps = 30
target_fps = 15
frame_indexes = list(range(0, 300, 10))
converted_indexes = convert_fps(frame_indexes, original_fps, target_fps)
```

### Visualization in Jupyter Notebooks
```python
from denku.visualization import show_image, show_images, show_video_in_jupyter, show_gif_in_jupyter

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
from denku.memory import empty_cuda_cache, print_cuda_allocated_memory

# Reset CUDA memory and run garbage collection
empty_cuda_cache()

# Print current CUDA memory usage
print_cuda_allocated_memory()
```


## Contacts
<p>Issues should be raised directly in the repository. For professional support and recommendations please <a>welcomedenk@gmail.com</a>.</p>