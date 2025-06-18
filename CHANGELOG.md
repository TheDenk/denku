# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.5]

### Added
- Exposed additional utility functions in main package:
  - Box operations: `get_boxes_intersection`, `get_boxes_union`, `get_boxes_iou`, `get_boxes_iou_matrix`
  - Image operations: `center_crop`
  - Video operations: `overlay_video`, `concatenate_videos`, `save_video_from_frames`
- Added comprehensive examples in README.md for all newly exposed functions

## [v0.1.4]

### Added
- Added `center_crop` and `rotate_image` functions

### Changed
- Refactored `show_images` to accept both titles (per-image subplot titles) and global_title (figure title) as separate, optional arguments

## [v0.1.3]

### Added
- Added `show_video_in_jupyter` function
- Added stride parameter to the `read_video` function

## [v0.1.2]

### Added
- Added `convert_video_fps` function to convert video frames between different frame rates
- Exposed additional utility functions in main package:
  - `resize_to_min_side`
  - `resize_to_max_side`
  - `create_video_grid`
- Enhanced documentation:
  - Added detailed usage examples for image processing
  - Added examples for video processing functions
  - Improved installation and development setup instructions
  - Added testing documentation

### Changed
- Fixed resize logic in `read_video` to maintain aspect ratio
- Changed default color space in `read_video` to RGB
- Improved documentation for max_side parameter in `read_video`
- Enhanced dependency management with separate requirements-dev.txt for development tools
- Improved project structure with proper setup.py configuration
- Added comprehensive test infrastructure with pytest

### Fixed
- Fixed test failures in test_utils.py:
  - Corrected test_split_on_chunks to handle numpy arrays properly
  - Updated test_get_linear_value and test_get_cosine_value test cases
  - Fixed test_get_ema_value assertions

## [v0.1.0]

### Added
- Major refactoring of the codebase into modular structure
- Split monolithic file into specialized modules:
  - utils: General utility functions
  - image: Image processing functions
  - visualization: Visualization and display functions
  - video: Video processing functions
  - memory: Memory management and PyTorch-related functions
- Added proper type hints and docstrings to all functions

### Changed
- Renamed `print_memory` to `print_cuda_allocated_memory` for clarity

## [v0.0.5]

### Added
- Added `mask2rle` and `rle2mask` functions
- Added `add_mask_on_image` function

### Fixed
- Fixed `split_on_chunks` method

## [v0.0.4]

### Added
- Added `show_gif_in_jupyter` function
- Added memory management functions:
  - `get_module_parameters_count_m`
  - `get_current_cuda_allocated_memory_gb`
  - `get_module_memory_gb`
- Added `get_capture_info` function
- Added `save_image` function

## [v0.0.3]

### Added
- Added value calculation functions:
  - `get_linear_value`
  - `get_cosine_value`
  - `get_ema_value`
- Added `show_video_in_jupyter` function

## [v0.0.1]

### Added
- Initial version
