"""
Transformer model architecture for music generation.
Includes positional encoding, attention mechanisms, and the main MusicTransformer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention between audio and text."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, key_padding_mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, key_len)
            scores = scores.masked_fill(mask, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        return self.w_o(context)


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and cross-attention."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention
        self.cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, text_context=None, text_mask=None, audio_mask=None):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, 
            key_padding_mask=audio_mask, 
            need_weights=False
        )
        x = x + self.dropout(attn_out)
        
        # Cross-attention with text (if provided)
        if text_context is not None:
            x_norm = self.norm2(x)
            cross_attn_out = self.cross_attn(x_norm, text_context, text_context, key_padding_mask=text_mask)
            x = x + self.dropout(cross_attn_out)
        
        # Feed-forward
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x


class MusicTransformer(nn.Module):
    """Transformer model for music generation with classifier-free guidance."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.n_codebooks = config['num_codebooks']
        self.vocab_size = config['vocab_size']
        self.max_seq_len = config['max_audio_seq_len']

        self.padding_idx = config['vocab_size'] - 1
        
        # Audio token embeddings for each codebook
        self.audio_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.padding_idx)
            for _ in range(self.n_codebooks)
        ])
        
        # Text projection layer
        self.text_projection = nn.Linear(config['text_embedding_dim'], self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_len)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                self.d_model, 
                self.n_heads, 
                config['d_ff'], 
                config['dropout']
            )
            for _ in range(self.n_layers)
        ])
        
        # Output heads for each codebook
        self.output_heads = nn.ModuleList([
            nn.Linear(self.d_model, self.vocab_size)
            for _ in range(self.n_codebooks)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config['dropout'])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        audio_tokens: torch.Tensor,
        text_embedding: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass of the transformer.
        
        Args:
            audio_tokens: [batch, n_codebooks, seq_len]
            text_embedding: [batch, text_seq_len, text_embedding_dim]
            text_attention_mask: [batch, text_seq_len] - True for valid tokens
            audio_attention_mask: [batch, seq_len] - True for valid tokens
        """
        batch_size, n_codebooks, seq_len = audio_tokens.shape
        device = audio_tokens.device

        # Embed audio tokens from each codebook and sum them
        audio_embeds = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        embedded = torch.stack([
            self.audio_embeddings[i](audio_tokens[:, i, :]) for i in range(n_codebooks)
        ], dim=0)
        
        audio_embeds = embedded.sum(dim=0)  # shape: (batch_size, seq_len, d_model)
        
        # Apply positional encoding
        audio_embeds = self.pos_encoding(audio_embeds)
        
        # Process text embedding
        text_context = None
        if text_embedding is not None:
            text_context = self.text_projection(text_embedding)
                
        # Pass through transformer layers
        hidden_states = audio_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                text_context=text_context,
                text_mask=(text_attention_mask == False) if text_attention_mask is not None else None,
                audio_mask=(audio_attention_mask == False) if audio_attention_mask is not None else None
            )
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Generate logits for each codebook
        logits = []
        for i in range(n_codebooks):
            logits.append(self.output_heads[i](hidden_states))
        
        # [batch, n_codebooks, seq_len, vocab_size]
        logits = torch.stack(logits, dim=1)
      
        return logits
