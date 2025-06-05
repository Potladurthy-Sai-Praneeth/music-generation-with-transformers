"""
Text and audio encoder classes for music generation.
Handles text embedding with FLAN-T5 and audio tokenization with EnCodec.
"""

import torch
import torch.nn as nn
import torchaudio
from transformers import AutoTokenizer, T5EncoderModel, EncodecModel
from typing import Tuple, Optional, Dict
import numpy as np


class TextEmbedder:
    """Text encoder using FLAN-T5 for generating text embeddings."""
    
    def __init__(self, model_name: str, max_length: int, device: str = "cpu"):
        """
        Initialize text embedder.
        
        Args:
            model_name: Name of the FLAN-T5 model
            max_length: Maximum sequence length for tokenization
            device: Device to run the model on
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
        self.max_length = max_length
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text inputs to embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Tuple of (embeddings, attention_mask)
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        return outputs.last_hidden_state, inputs["attention_mask"]


class AudioCodec:
    """Audio codec using EnCodec for audio tokenization and reconstruction."""
    
    def __init__(self, model_name: str = "facebook/encodec_24khz", config: dict = None, device: str = "cpu"):
        """
        Initialize audio codec.
        
        Args:
            model_name: Name of the EnCodec model
            config: Configuration dictionary
            device: Device to run the model on
        """
        self.device = torch.device(device)
        self.config = config or {}
        self.model = EncodecModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()
        
        self.sample_rate = self.config.get('target_audio_sample_rate', 24000)
        self.channels = self.model.config.audio_channels
        
        self.num_codebooks = self.config.get('num_codebooks', 2)
        self.codebook_size = getattr(self.model.config, 'codebook_size', self.config.get('vocab_size', 1024))
        
        # Use the last token in codebook as pad token
        self.pad_token_value = self.codebook_size - 1
    
    @torch.no_grad()
    def encode_audio(self, audio_path: str, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert audio file to discrete tokens using EnCodec.
        
        Args:
            audio_path: Path to the audio file
            normalize: Whether to normalize audio amplitude
            
        Returns:
            Tuple of (tokens, attention_mask)
        """
        try:
            wav, original_sr = torchaudio.load(audio_path)
            
            if original_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(original_sr, self.sample_rate)
                wav = resampler(wav)
            
            if normalize:
                wav = wav / (wav.abs().max() + 1e-8)
            
            wav = wav.to(self.device)
            
            # Ensure correct dimensions: [batch, channels, time]
            if wav.dim() == 1:
                wav = wav.unsqueeze(0).unsqueeze(0)
            elif wav.dim() == 2:
                wav = wav.unsqueeze(0)

            # Encode audio to tokens
            with torch.no_grad():
                encoded_frames = self.model.encode(wav)

            # Extract codes (tokens)
            codes = encoded_frames.audio_codes.squeeze(1)  # [batch, n_codebooks, sequence_length]
            
            batch_size, num_codebooks, current_seq_len = codes.shape
            max_seq_len = self.config.get("max_audio_seq_len", current_seq_len)
            
            if current_seq_len > max_seq_len:
                codes = codes[:, :, :max_seq_len]
                current_seq_len = max_seq_len

            # Create fixed-size output with proper padding
            padded_tokens = torch.full(
                (batch_size, num_codebooks, max_seq_len),
                self.pad_token_value,
                dtype=codes.dtype,
                device=codes.device
            )
            
            # Create attention mask
            attention_mask = torch.zeros(
                (batch_size, max_seq_len),
                dtype=torch.bool,
                device=codes.device
            )

            padded_tokens[:, :, :current_seq_len] = codes
            attention_mask[:, :current_seq_len] = True
            
            return padded_tokens, attention_mask
            
        except Exception as e:
            raise e
    
    def decode_tokens(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convert discrete tokens back to audio using EnCodec decoder.
        
        Args:
            tokens: Discrete tokens of shape [batch, n_codebooks, sequence_length]
            attention_mask: Mask indicating real vs padded tokens
            
        Returns:
            Reconstructed audio waveform [batch, channels, time]
        """
        try:
            if tokens.dim() != 3:
                raise ValueError(f"Expected tokens with 3 dimensions [batch, n_codebooks, seq_len], got {tokens.shape}")
            
            tokens = tokens.to(self.device)
            batch_size, num_codebooks, seq_len = tokens.shape
            
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                actual_lengths = attention_mask.sum(dim=1)
                max_actual_length = actual_lengths.max().item()
                if max_actual_length < seq_len:
                    tokens = tokens[:, :, :max_actual_length]
            
            # Validate codebook dimension
            expected_codebooks = self.config.get('num_codebooks', 2)
            assert num_codebooks == expected_codebooks, \
                f'Expected {expected_codebooks} codebooks, got {num_codebooks}'

            with torch.no_grad():
                # Add extra dimension to match expected input shape
                audio_values = self.model.decode(tokens.unsqueeze(1), audio_scales=[None])
            
            if hasattr(audio_values, 'audio_values'):
                decoded_audio = audio_values.audio_values
            else:
                decoded_audio = audio_values
                
            if decoded_audio.dim() == 2:
                decoded_audio = decoded_audio.unsqueeze(0)
                
            return decoded_audio
            
        except Exception as e:
            raise e
