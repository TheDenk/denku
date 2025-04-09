"""Memory management utilities for the denku package."""

import gc
import torch
from typing import Union


def get_module_parameters_count_m(module: torch.nn.Module) -> float:
    """Get number of parameters in millions.
    
    Args:
        module (torch.nn.Module): PyTorch module
        
    Returns:
        float: Number of parameters in millions
    """
    return sum(p.numel() for p in module.parameters()) / 1e6


def get_current_cuda_allocated_memory_gb() -> float:
    """Get current CUDA allocated memory in GB.
    
    Returns:
        float: Allocated memory in GB
    """
    return torch.cuda.memory_allocated() / 1e9


def get_module_memory_gb(module: torch.nn.Module, dtype: str = 'fp32') -> float:
    """Get module memory usage in GB.
    
    Args:
        module (torch.nn.Module): PyTorch module
        dtype (str, optional): Data type. Defaults to 'fp32'.
        
    Returns:
        float: Memory usage in GB
    """
    param_size = 0
    for param in module.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in module.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    if dtype == 'fp16':
        size_all_mb /= 2
    return size_all_mb / 1024


def log_trainable_params(logger: Any, model: torch.nn.Module,
                        model_name: str) -> None:
    """Log trainable parameters information.
    
    Args:
        logger: Logger object
        model (torch.nn.Module): PyTorch module
        model_name (str): Model name
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'{model_name} total params: {total_params:,}')
    logger.info(f'{model_name} trainable params: {trainable_params:,}')


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """Print trainable parameters information.
    
    Args:
        model (torch.nn.Module): PyTorch module
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}')
    print(f'Trainable params: {trainable_params:,}')


def empty_cuda_cache() -> None:
    """Reset CUDA memory and garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def print_cuda_allocated_memory() -> None:
    """Print current memory usage."""
    if torch.cuda.is_available():
        print(f'Allocated: {get_current_cuda_allocated_memory_gb():.2f} GB')
        print(f'Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB')
