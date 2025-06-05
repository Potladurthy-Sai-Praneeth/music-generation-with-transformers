"""
Inference for music generation with classifier-free guidance.
Handles token generation, audio conversion, and visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Union
from IPython.display import Audio, display

from model import MusicTransformer
from encoders import TextEmbedder, AudioCodec


class MusicGenerationInference:
    """Inference class for music generation with classifier-free guidance."""
    
    def __init__(self, model: MusicTransformer, config: dict, text_embedder: TextEmbedder, 
                 audio_codec: AudioCodec, device: str):
        """
        Initialize inference engine.
        
        Args:
            model: Trained MusicTransformer model
            config: Configuration dictionary
            text_embedder: Text embedding model
            audio_codec: Audio codec for token conversion
            device: Device to run inference on
        """
        self.model = model
        self.config = config
        self.text_embedder = text_embedder
        self.audio_codec = audio_codec
        self.device = device
        
        self.model.eval()
        
        # Inference parameters
        self.cfg_weight = config.get('cfg_weight', 3.0)
        self.temperature = config.get('temperature', 1.0)
        self.top_k = config.get('top_k', 0)
        self.top_p = config.get('top_p', 0.9)
        
        # Create output directories
        self.output_dir = Path("generated_music")
        self.output_dir.mkdir(exist_ok=True)
        
        # Conditional output directories
        self.conditional_dir = self.output_dir / "conditional"
        self.conditional_dir.mkdir(exist_ok=True)
        (self.conditional_dir / "audio").mkdir(exist_ok=True)
        (self.conditional_dir / "waveforms").mkdir(exist_ok=True)
        (self.conditional_dir / "spectrograms").mkdir(exist_ok=True)
        
        # Unconditional output directories
        self.unconditional_dir = self.output_dir / "unconditional"
        self.unconditional_dir.mkdir(exist_ok=True)
        (self.unconditional_dir / "audio").mkdir(exist_ok=True)
        (self.unconditional_dir / "waveforms").mkdir(exist_ok=True)
        (self.unconditional_dir / "spectrograms").mkdir(exist_ok=True)
    
    def get_output_directories(self, is_conditional: bool):
        """Get the appropriate output directories based on generation type."""
        base_dir = self.conditional_dir if is_conditional else self.unconditional_dir
        return {
            'audio': base_dir / "audio",
            'waveforms': base_dir / "waveforms", 
            'spectrograms': base_dir / "spectrograms"
        }
    
    def sample_from_logits(self, logits: torch.Tensor, temperature: float = 1.0, 
                          top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
        """Sample tokens from logits with temperature, top-k, and top-p sampling."""
        logits = logits / temperature
        
        # Top-k sampling
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, top_k_indices, top_k_logits)
        
        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -1e9
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, 1)
        
        return sampled_tokens.squeeze(-1)
    
    @torch.no_grad()
    def generate_tokens_with_cfg(self, text_prompt: Optional[str] = None, max_length: int = 750, 
                                cfg_weight: float = 3.0, use_prompt: bool = True,
                                text_embedding=None, text_attention_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate audio tokens using classifier-free guidance.
        
        Args:
            text_prompt: Text prompt for conditional generation
            max_length: Maximum sequence length to generate
            cfg_weight: CFG weight (higher = more text conditioning)
            use_prompt: Whether to use the text prompt
            text_embedding: Pre-computed text embedding
            text_attention_mask: Pre-computed text attention mask
            
        Returns:
            Tuple of (generated_tokens, attention_mask)
        """
        if max_length is None:
            max_length = self.config['max_audio_seq_len']
        if cfg_weight is None:
            cfg_weight = self.cfg_weight
            
        batch_size = 1
        n_codebooks = self.config['num_codebooks']
        vocab_size = self.config['vocab_size']
        
        # Prepare text embeddings
        if not use_prompt and text_embedding is not None and text_attention_mask is not None:
            text_embedding = text_embedding.to(self.device)
            text_attention_mask = text_attention_mask.to(self.device)
        elif text_prompt is not None and use_prompt:
            text_embedding, text_attention_mask = self.text_embedder.encode([text_prompt])
            text_embedding = text_embedding.to(self.device)
            text_attention_mask = text_attention_mask.to(self.device)
        elif text_prompt is None and use_prompt:
            # For unconditional generation, use zero embeddings
            text_embedding = torch.zeros(1, 1, self.config['text_embedding_dim']).to(self.device)
            text_attention_mask = torch.zeros(1, 1, dtype=torch.bool).to(self.device)
        
        # Initialize with random start tokens
        generated_tokens = torch.randint(0, vocab_size-1, (batch_size, n_codebooks, 1), dtype=torch.long).to(self.device)
        
        print(f"Generating {'conditional' if text_prompt else 'unconditional'} audio...")
        print(f"Text prompt: {text_prompt if text_prompt else 'None'}")
        print(f"CFG weight: {cfg_weight}")
        
        # Generate tokens autoregressively
        for step in tqdm(range(max_length - 1), desc="Generating tokens"):
            current_seq_len = generated_tokens.shape[-1]
            
            # Create attention mask for current sequence
            audio_attention_mask = torch.ones(batch_size, current_seq_len, dtype=torch.bool).to(self.device)
            
            if text_prompt is not None and cfg_weight > 1.0:
                # Classifier-Free Guidance: compute both conditional and unconditional logits
                
                # Conditional forward pass
                conditional_logits = self.model(
                    generated_tokens,
                    text_embedding=text_embedding,
                    text_attention_mask=text_attention_mask,
                    audio_attention_mask=audio_attention_mask
                )
                
                # Unconditional forward pass (zero text embeddings)
                zero_text_embedding = torch.zeros_like(text_embedding)
                zero_text_mask = torch.zeros_like(text_attention_mask)
                
                unconditional_logits = self.model(
                    generated_tokens,
                    text_embedding=zero_text_embedding,
                    text_attention_mask=zero_text_mask,
                    audio_attention_mask=audio_attention_mask
                )
                
                # Apply CFG
                logits = unconditional_logits + cfg_weight * (conditional_logits - unconditional_logits)
                
            else:
                # Standard forward pass (conditional or unconditional based on text_prompt)
                logits = self.model(
                    generated_tokens,
                    text_embedding=text_embedding,
                    text_attention_mask=text_attention_mask,
                    audio_attention_mask=audio_attention_mask
                )
            
            # Get logits for the last position
            next_token_logits = logits[:, :, -1, :]  # [batch, n_codebooks, vocab_size]
            
            # Sample next tokens for each codebook
            next_tokens = torch.zeros(batch_size, n_codebooks, 1, dtype=torch.long).to(self.device)
            
            for codebook_idx in range(n_codebooks):
                sampled_token = self.sample_from_logits(
                    next_token_logits[:, codebook_idx, :],
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p
                )
                next_tokens[:, codebook_idx, 0] = sampled_token
            
            # Append to generated sequence
            generated_tokens = torch.cat([generated_tokens, next_tokens], dim=-1)
            
            # Early stopping if we hit padding tokens for all codebooks
            if (next_tokens == (vocab_size - 1)).all():
                break
        
        # Create final attention mask
        final_attention_mask = torch.ones(batch_size, generated_tokens.shape[-1], dtype=torch.bool).to(self.device)
        
        return generated_tokens, final_attention_mask
    
    def tokens_to_audio(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Convert tokens back to audio waveform."""
        try:
            audio_waveform = self.audio_codec.decode_tokens(tokens, attention_mask)
            return audio_waveform.cpu()
        except Exception as e:
            print(f"Error decoding tokens to audio: {e}")
            raise e
    
    def save_audio(self, waveform: torch.Tensor, filename: str, is_conditional: bool, sample_rate: Optional[int] = None):
        """Save audio waveform to file in appropriate directory."""
        if sample_rate is None:
            sample_rate = self.config['target_audio_sample_rate']
        
        # Ensure waveform is in correct format
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)  # Remove batch dimension
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)  # Remove channel dimension if mono
        
        # Convert to numpy and ensure float32 dtype
        waveform_np = waveform.numpy()
        if waveform_np.dtype == np.float16:
            waveform_np = waveform_np.astype(np.float32)
        elif waveform_np.dtype not in [np.float32, np.float64, np.int16, np.int32]:
            waveform_np = waveform_np.astype(np.float32)
        
        # Normalize audio to prevent clipping
        if waveform_np.dtype in [np.float32, np.float64]:
            max_val = np.abs(waveform_np).max()
            if max_val > 1.0:
                waveform_np = waveform_np / max_val
        
        # Get appropriate directory
        dirs = self.get_output_directories(is_conditional)
        audio_path = dirs['audio'] / f"{filename}.wav"
                
        sf.write(audio_path, waveform_np.T if waveform_np.ndim > 1 else waveform_np, sample_rate)
        print(f"Audio saved: {audio_path}")
        return audio_path
    
    def display_audio(self, waveform: torch.Tensor, sample_rate: Optional[int] = None):
        """Display audio player in notebook."""
        if sample_rate is None:
            sample_rate = self.config['target_audio_sample_rate']
        
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
            
        waveform_np = waveform.numpy()
        return Audio(waveform_np, rate=sample_rate)
    
    def plot_waveform(self, waveform: torch.Tensor, filename: str, is_conditional: bool, sample_rate: Optional[int] = None):
        """Plot and save waveform visualization."""
        if sample_rate is None:
            sample_rate = self.config['target_audio_sample_rate']
        
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
            
        waveform_np = waveform.numpy()
        
        plt.figure(figsize=(12, 4))
        time_axis = np.linspace(0, len(waveform_np) / sample_rate, len(waveform_np))
        plt.plot(time_axis, waveform_np)
        plt.title(f'Waveform: {filename} ({"Conditional" if is_conditional else "Unconditional"})')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Get appropriate directory
        dirs = self.get_output_directories(is_conditional)
        waveform_path = dirs['waveforms'] / f"{filename}_waveform.png"
        
        plt.savefig(waveform_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Waveform plot saved: {waveform_path}")
    
    def plot_spectrogram(self, waveform: torch.Tensor, filename: str, is_conditional: bool, sample_rate: Optional[int] = None):
        """Plot and save spectrogram visualization."""
        if sample_rate is None:
            sample_rate = self.config['target_audio_sample_rate']
        
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
            
        waveform_np = waveform.numpy()
        if waveform_np.dtype == np.float16:
            waveform_np = waveform_np.astype(np.float32)
        
        plt.figure(figsize=(12, 6))
        
        # Compute spectrogram
        D = librosa.stft(waveform_np, n_fft=2048, hop_length=512)
        DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        librosa.display.specshow(DB, sr=sample_rate, hop_length=512, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {filename} ({"Conditional" if is_conditional else "Unconditional"})')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        
        # Get appropriate directory
        dirs = self.get_output_directories(is_conditional)
        spectrogram_path = dirs['spectrograms'] / f"{filename}_spectrogram.png"
        
        plt.savefig(spectrogram_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Spectrogram plot saved: {spectrogram_path}")
    
    def generate_and_save(self, text_prompt: Optional[str] = None, 
                         filename_prefix: str = "generated", 
                         cfg_weight: float = None,
                         max_length: int = None) -> Tuple[torch.Tensor, str]:
        """
        Complete generation pipeline: generate, save, and visualize.
        
        Returns:
            Tuple of (waveform, audio_filename)
        """
        is_conditional = text_prompt is not None
        
        print("="*80)
        print(f"MUSIC GENERATION - {'CONDITIONAL' if is_conditional else 'UNCONDITIONAL'}")
        print("="*80)
        
        # Generate tokens
        tokens, attention_mask = self.generate_tokens_with_cfg(
            text_prompt=text_prompt,
            max_length=max_length,
            cfg_weight=cfg_weight
        )
        
        print(f"Generated tokens shape: {tokens.shape}")
        
        # Convert to audio
        print("Converting tokens to audio...")
        waveform = self.tokens_to_audio(tokens, attention_mask)
        
        if waveform is None:
            raise ValueError('Failed to generate audio')
        
        # Create filename
        if is_conditional:
            safe_prompt = "".join(c for c in text_prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')[:20]  # Limit length
            filename = f"{filename_prefix}_{safe_prompt}"
        else:
            filename = f"{filename_prefix}_unconditional"
        
        # Save audio
        audio_path = self.save_audio(waveform, filename, is_conditional)
        
        # Display audio player (if in notebook environment)
        try:
            print(f"\nGenerated {'Conditional' if is_conditional else 'Unconditional'} Audio:")
            audio_widget = self.display_audio(waveform)
            display(audio_widget)
        except:
            print("Audio display not available (not in notebook environment)")
        
        # Plot waveform
        print("\nGenerating waveform plot...")
        self.plot_waveform(waveform, filename, is_conditional)
        
        # Plot spectrogram
        print("\nGenerating spectrogram...")
        self.plot_spectrogram(waveform, filename, is_conditional)
        
        print(f"\nGeneration complete! Files saved in {'conditional' if is_conditional else 'unconditional'} directory")
        print(f"Filename: {filename}")
        print("="*80)
        
        return waveform, filename
