"""
Simple example script for training and inference.
This demonstrates basic usage of the music generation system.
"""

import torch
import os
from config import CONFIG
from main import train_model, run_inference
from utils import set_seed, create_directories, get_device


def quick_train_example():
    """Example of how to train the model with a small dataset."""
    print("="*60)
    print("QUICK TRAINING EXAMPLE")
    print("="*60)
    
    # Set up configuration for quick training
    config = CONFIG.copy()
    config['dataset_size'] = 1000  # Small dataset for testing
    config['num_epochs'] = 2       # Just 2 epochs for demo
    config['batch_size'] = 4       # Small batch size
    config['device'] = get_device()
    
    set_seed(42)
    create_directories(config)
    
    print("Training with reduced dataset size for demonstration...")
    print(f"Dataset size: {config['dataset_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    
    # Note: This would require the actual MusicBench dataset to be available
    # For demonstration purposes, this shows the structure
    print("\nNote: This example requires the MusicBench dataset to be available.")
    print("Make sure to set the correct dataset_path in config.py")


def quick_inference_example():
    """Example of how to run inference (requires a trained model)."""
    print("="*60)
    print("QUICK INFERENCE EXAMPLE")
    print("="*60)
    
    checkpoint_path = "checkpoints/checkpoint_epoch_50.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        print("Train the model first or provide a valid checkpoint path.")
        return
    
    config = CONFIG.copy()
    config['device'] = get_device()
    
    set_seed(42)
    create_directories(config)
    
    print("Running inference with example prompts...")
    
    # Example prompts
    example_prompts = [
        "A cheerful piano melody",
        "Dark ambient electronic music",
        "Classical orchestra piece"
    ]
    
    # This would run the actual inference
    print("Example prompts:")
    for i, prompt in enumerate(example_prompts):
        print(f"  {i+1}. {prompt}")
    
    print("\nNote: This example requires a trained model checkpoint.")
    print("Train the model first using the training example.")


def main():
    """Run examples."""
    print("Music Generation Examples")
    print("=" * 60)
    print("1. Quick training example")
    print("2. Quick inference example")
    print("=" * 60)
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        quick_train_example()
    elif choice == "2":
        quick_inference_example()
    else:
        print("Invalid choice. Please run again and select 1 or 2.")


if __name__ == "__main__":
    main()
