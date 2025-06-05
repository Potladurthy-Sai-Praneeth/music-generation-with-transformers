"""
Training module for music generation with classifier-free guidance.
Handles model training, validation, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Dict, Tuple, Optional, Any
from model import MusicTransformer


class MusicGenerationTrainer:
    """Trainer class for music generation with CFG."""

    def __init__(self, model: MusicTransformer, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            model: MusicTransformer model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.text_dropout_prob = config.get('cfg_dropout_rate')
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'] * config.get('steps_per_epoch', 100),
            eta_min=config['learning_rate'] * 0.1
        )
        
        # CFG parameters
        self.cfg_weight = config.get('cfg_weight', 3.0)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.config['vocab_size'] - 1)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss with classifier-free guidance."""
        
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        audio_tokens = batch['audio_tokens']  # [batch, n_codebooks, seq_len]
        text_embedding = batch['text_embedding']  # [batch, text_seq_len, embed_dim]
        text_attention_mask = batch['text_attention_mask']  # [batch, text_seq_len]
        cfg_mask = batch['cfg_mask']  # [batch, 2] -> [text_dropped, audio_dropped]
        audio_attention_mask = batch['audio_attention_mask']
        
        batch_size, n_codebooks, seq_len = audio_tokens.shape
        
        # Prepare input and target tokens
        input_tokens = audio_tokens[:, :, :-1]  # Remove last token for input
        target_tokens = audio_tokens[:, :, 1:]   # Remove first token for target
        
        # Handle CFG masking
        text_input = text_embedding.clone()
        text_mask_input = text_attention_mask.clone()
        
        # Zero out text where CFG mask indicates text dropout
        text_dropped = cfg_mask[:, 0]  # [batch]
        if text_dropped.any():
            text_input[text_dropped] = 0
            text_mask_input[text_dropped] = 0

        # Forward pass
        logits = self.model(
            input_tokens,
            text_embedding=text_input,
            text_attention_mask=text_mask_input,
            audio_attention_mask=audio_attention_mask[:, :-1]
        )   # [batch, n_codebooks, seq_len-1, vocab_size]        # Compute loss for each codebook
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        losses = {}
        
        for i in range(logits.shape[1]):
            logits_i = logits[:, i, :, :].reshape(-1, logits.shape[-1])  # [batch*(seq_len-1), vocab_size]
            targets_i = target_tokens[:, i, :].reshape(-1)  # [batch*(seq_len-1)]
            
            loss_i = self.criterion(logits_i, targets_i) * 10
            total_loss = total_loss + loss_i
            losses[f'loss_codebook_{i}'] = loss_i.item()
        
        # Average loss across codebooks
        total_loss = total_loss / n_codebooks
        losses['total_loss'] = total_loss.item()
        
        return total_loss, losses
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, losses = self.compute_loss(batch)
        
        loss.backward()
        
        if self.config.get('gradient_clip_norm'):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['gradient_clip_norm']
            )
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        losses['learning_rate'] = self.scheduler.get_last_lr()[0]
        
        return losses
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Validation step."""
        self.model.eval()
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                _, losses = self.compute_loss(batch)
                
                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0
                    total_losses[key] += value
                
                num_batches += 1
        
        avg_losses = {f"val_{key}": value / num_batches for key, value in total_losses.items()}
        return avg_losses
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Training loop."""
        
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            
            # Training
            self.model.train()
            epoch_losses = {}
            num_batches = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            for batch in progress_bar:
                losses = self.train_step(batch)
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value
                
                num_batches += 1
                progress_bar.set_postfix({
                    'loss': f"{losses['total_loss']:.4f}",
                    'lr': f"{losses['learning_rate']:.2e}"
                })
                            
            avg_train_losses = {key: value / num_batches for key, value in epoch_losses.items()}
            
            if val_dataloader is not None:
                val_losses = self.validate(val_dataloader)
                avg_train_losses.update(val_losses)
            
            print(f"\nEpoch {epoch+1} Summary:")
            for key, value in avg_train_losses.items():
                print(f"  {key}: {value:.4f}")
                
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(epoch + 1)
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")
