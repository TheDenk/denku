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
