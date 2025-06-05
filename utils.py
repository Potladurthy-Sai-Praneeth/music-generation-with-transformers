"""
Utility functions for music generation project.
"""

import os
import torch
import random
import numpy as np
from pathlib import Path


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(config: dict):
    """Create necessary directories for training and inference."""
    dirs_to_create = [
        config.get('cache_dir', 'pre-processed-music-bench'),
        config.get('checkpoint_dir', 'checkpoints'),
        'generated_music',
        'generated_music/conditional/audio',
        'generated_music/conditional/waveforms',
        'generated_music/conditional/spectrograms',
        'generated_music/unconditional/audio',
        'generated_music/unconditional/waveforms',
        'generated_music/unconditional/spectrograms',
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("Using CPU device")
    return device


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """Format time in seconds to human readable format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    if hours > 0:
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    else:
        return f"{int(minutes):02d}:{int(seconds):02d}"


def load_prompts_from_file(file_path: str):
    """Load text prompts from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompts file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    return prompts


def save_prompts_to_file(prompts: list, file_path: str):
    """Save text prompts to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")


def get_default_prompts():
    """Get a list of default prompts for testing."""
    return [
        "A cheerful piano melody",
        "Dark ambient electronic music with synthesizers",
        "Classical orchestra piece with strings and woodwinds",
        "Electric guitar with high tempo and C Major",
        "Experimental ambient techno with found sounds, granular synthesis, and evolving textures"
    ]
