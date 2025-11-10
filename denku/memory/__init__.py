"""Memory management utilities for the denku package."""

import gc
import torch
from typing import Any, Union


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

def print_model_info(*models, model_names=None):
    """
    Print model parameters info.
    Args:
        *models: PyTorch models.
        model_names [optional]: Model names for print (same len as models).
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(models))]
    elif len(model_names) != len(models):
        raise ValueError("model_names should be the same amount as models.")

    all_data = []
    for model, name in zip(models, model_names):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        dtype = next(model.parameters()).dtype
        bytes_per_param = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}.get(dtype, 4)
        size_gb = total_params * bytes_per_param / (1024**3)
        
        all_data.append({
            "name": name,
            "dtype": str(dtype),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "size_gb": size_gb
        })
    
    _print_combined_model_table(all_data)

def _print_combined_model_table(models_data):
    headers = ["Model", "Data type", "Total params", "Trainable", "Frozen", "Memory size"]
    
    table_data = []
    for data in models_data:
        row = [
            data["name"],
            data["dtype"],
            f"{data['total_params'] / 1e6:.1f} M",
            f"{data['trainable_params'] / 1e6:.1f} M",
            f"{data['frozen_params'] / 1e6:.1f} M",
            f"{data['size_gb']:.2f} GB"
        ]
        table_data.append(row)
    
    col_widths = []
    for i in range(len(headers)):
        max_width = max(len(str(row[i])) for row in table_data)
        max_width = max(max_width, len(headers[i]))
        col_widths.append(max_width + 2)  
    
    total_width = sum(col_widths) + len(headers) + 7
    
    print("\n" + "=" * total_width)
    print("│", end="")
    for header, width in zip(headers, col_widths):
        print(f" {header:^{width}}│", end="")
    print("\n" + "=" * total_width)
    
    for row in table_data:
        print("│", end="")
        for value, width in zip(row, col_widths):
            print(f" {value:<{width}}│", end="")
        print()
    
    print("=" * total_width)
    total_params_all = sum(data["total_params"] for data in models_data)
    trainable_params_all = sum(data["trainable_params"] for data in models_data)
    
    print(f"Total params: {total_params_all / 1e6:.1f} M.")
    print(f"Trainable params: {trainable_params_all / 1e6:.1f} M / {trainable_params_all/total_params_all*100:.1f}% of total.")
    
