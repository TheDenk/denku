v0.1.2
-------
- Added convert_video_fps function to convert video frames between different frame rates
- Fixed resize logic in read_video to maintain aspect ratio
- Changed default color space in read_video to RGB
- Improved documentation for max_side parameter in read_video
- Enhanced dependency management with separate requirements-dev.txt for development tools
- Improved project structure with proper setup.py configuration
- Added comprehensive test infrastructure with pytest
- Fixed test failures in test_utils.py:
  - Corrected test_split_on_chunks to handle numpy arrays properly
  - Updated test_get_linear_value and test_get_cosine_value test cases
  - Fixed test_get_ema_value assertions
- Exposed additional utility functions in main package:
  - resize_to_min_side
  - resize_to_max_side
  - create_video_grid
- Enhanced documentation:
  - Added detailed usage examples for image processing
  - Added examples for video processing functions
  - Improved installation and development setup instructions
  - Added testing documentation

v0.1.0
-------
- Major refactoring of the codebase into modular structure
- Split monolithic file into specialized modules:
  - utils: General utility functions
  - image: Image processing functions
  - visualization: Visualization and display functions
  - video: Video processing functions
  - memory: Memory management and PyTorch-related functions
- Added proper type hints and docstrings to all functions
- Renamed print_memory to print_cuda_allocated_memory for clarity

v0.0.5
-------
- fixed split_on_chunks method
- mask2rle, rle2mask
- add_mask_on_image


v0.0.4
-------
- show_gif_in_jupyter, get_module_parameters_count_m, get_current_cuda_allocated_memory_gb, get_module_memory_gb, get_capture_info, save_image


v0.0.3
-------
- get_linear_value, get_cosine_value, get_ema_value, show_video_in_jupyter


v0.0.1
-------
- initial version
