"""
Main script for training and inference of music generation model.
Supports both training from scratch and inference with pre-trained models.
"""

import argparse
import torch
from torch.utils.data import DataLoader
import os
import sys

from config import CONFIG
from dataset import OptimizedMusicBenchDataset, optimized_collate_fn
from encoders import TextEmbedder, AudioCodec
from model import MusicTransformer
from trainer import MusicGenerationTrainer
from inference import MusicGenerationInference
from utils import set_seed, create_directories, get_device, count_parameters, get_default_prompts, load_prompts_from_file


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Music Generation with Transformer and CFG')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True,
                       help='Mode to run: train or inference')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--dataset_size', type=int, default=None,
                       help='Size of dataset subset to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    # Inference arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint for inference')
    parser.add_argument('--prompt', type=str, default='',
                       help='Text prompt for conditional generation')
    parser.add_argument('--batch_prompts', type=str, default=None,
                       help='Path to file containing multiple prompts')
    parser.add_argument('--num_unconditional', type=int, default=5,
                       help='Number of unconditional samples to generate')
    parser.add_argument('--cfg_weight', type=float, default=None,
                       help='CFG weight for inference')
    parser.add_argument('--max_length', type=int, default=None,
                       help='Maximum generation length')
    parser.add_argument('--output_prefix', type=str, default='generated',
                       help='Prefix for output filenames')
    
    # General arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """Update configuration with command line arguments."""
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.dataset_size is not None:
        config['dataset_size'] = args.dataset_size
    if args.cfg_weight is not None:
        config['cfg_weight'] = args.cfg_weight
    if args.max_length is not None:
        config['max_audio_seq_len'] = args.max_length
    if args.device is not None:
        config['device'] = args.device
    else:
        config['device'] = get_device()
    
    return config


def train_model(config, args):
    """Train the music generation model."""
    print("="*80)
    print("TRAINING MODE")
    print("="*80)
    
    device = config['device']
    
    # Create datasets
    print("Loading training dataset...")
    train_dataset = OptimizedMusicBenchDataset(config, device, 'dataset_split_train')
    
    print("Loading validation dataset...")
    val_dataset = OptimizedMusicBenchDataset(config, device, 'dataset_split_eval')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=optimized_collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=optimized_collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    model = MusicTransformer(config).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Initialize trainer
    trainer = MusicGenerationTrainer(model, config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting training...")
    trainer.train(train_loader, val_loader)
    
    print("Training completed!")


def run_inference(config, args):
    """Run inference with the trained model."""
    print("="*80)
    print("INFERENCE MODE")
    print("="*80)
    
    if args.checkpoint is None:
        print("Error: --checkpoint is required for inference mode")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    device = config['device']
    
    # Initialize components
    print("Initializing text embedder...")
    text_embedder = TextEmbedder(config["text_encoder_model_name"], config["max_prompt_length"], device)
    
    print("Initializing audio codec...")
    audio_codec = AudioCodec(config["audio_codec_model_name"], config, device)
    
    print("Initializing model...")
    model = MusicTransformer(config).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize inference engine
    inference_engine = MusicGenerationInference(
        model=model,
        config=config,
        text_embedder=text_embedder,
        audio_codec=audio_codec,
        device=device
    )
    
    # Generate music based on arguments
    if args.batch_prompts:
        # Generate from file of prompts
        print(f"Loading prompts from: {args.batch_prompts}")
        prompts = load_prompts_from_file(args.batch_prompts)
        
        for i, prompt in enumerate(prompts):
            print(f"\nGenerating sample {i+1}/{len(prompts)}")
            try:
                inference_engine.generate_and_save(
                    text_prompt=prompt,
                    filename_prefix=f"{args.output_prefix}_conditional_{i+1}",
                    cfg_weight=config['cfg_weight'],
                    max_length=config['max_audio_seq_len']
                )
            except Exception as e:
                print(f"Error generating for prompt '{prompt}': {e}")
                
    elif args.prompt:
        # Generate from single prompt
        print(f"Generating conditional sample with prompt: '{args.prompt}'")
        inference_engine.generate_and_save(
            text_prompt=args.prompt,
            filename_prefix=f"{args.output_prefix}_conditional",
            cfg_weight=config['cfg_weight'],
            max_length=config['max_audio_seq_len']
        )
        
    else:
        # Generate default samples
        print("Generating default conditional samples...")
        default_prompts = get_default_prompts()[:5]  # Use first 5 default prompts
        
        for i, prompt in enumerate(default_prompts):
            print(f"\nGenerating conditional sample {i+1}/{len(default_prompts)}")
            try:
                inference_engine.generate_and_save(
                    text_prompt=prompt,
                    filename_prefix=f"{args.output_prefix}_conditional_{i+1}",
                    cfg_weight=config['cfg_weight'],
                    max_length=config['max_audio_seq_len']
                )
            except Exception as e:
                print(f"Error generating for prompt '{prompt}': {e}")
    
        # Generate unconditional samples
        print(f"\nGenerating {args.num_unconditional} unconditional samples...")
        for i in range(args.num_unconditional):
            print(f"\nGenerating unconditional sample {i+1}/{args.num_unconditional}")
            try:
                inference_engine.generate_and_save(
                    text_prompt=None,
                    filename_prefix=f"{args.output_prefix}_unconditional_{i+1}",
                    cfg_weight=config['cfg_weight'],
                    max_length=config['max_audio_seq_len']
                )
            except Exception as e:
                print(f"Error generating unconditional sample {i+1}: {e}")
    
    print("\nInference completed!")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Update config with command line arguments
    config = update_config_from_args(CONFIG.copy(), args)
    
    # Create necessary directories
    create_directories(config)
    
    # Run appropriate mode
    if args.mode == 'train':
        train_model(config, args)
    elif args.mode == 'inference':
        run_inference(config, args)


if __name__ == "__main__":
    main()
