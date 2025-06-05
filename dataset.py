"""
Dataset loading and preprocessing for music generation.
Handles MusicBench dataset with caching and optimized batching.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import hashlib
import pickle
from tqdm import tqdm
from datasets import load_dataset
from typing import Dict, Any, List, Tuple
from encoders import TextEmbedder, AudioCodec


class OptimizedMusicBenchDataset(Dataset):
    """Optimized dataset class for MusicBench with caching and CFG support."""
    
    def __init__(self, config: Dict[str, Any], device, split_type='dataset_split_train'):
        """
        Initialize dataset with caching support.
        
        Args:
            config: Configuration dictionary
            device: Device to use for processing
            split_type: Type of dataset split ('dataset_split_train' or 'dataset_split_eval')
        """
        print(f"Loading Hugging Face dataset: {config['dataset_name']}")
        
        self.config = config
        self.device = device
        self.split_name = config[split_type]
        
        # Create cache directory
        self.cache_dir = config.get("cache_dir", "./dataset_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Generate cache key based on config
        cache_key = self._generate_cache_key(config)
        self.cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # CFG parameters
        self.cfg_dropout_prob = config.get("cfg_dropout_rate", 0.1)
        self.max_audio_length = config.get("max_audio_duration_sec")
        
        # Load or create preprocessed dataset
        if os.path.exists(self.cache_file) and not config.get("force_preprocess", False):
            print(f"Loading cached dataset from {self.cache_file}")
            self.preprocessed_data = self._load_cache()
        else:
            print("Preprocessing dataset...")
            self.preprocessed_data = self._preprocess_dataset(config, device)
            self._save_cache()
        
        print(f"Dataset ready: {len(self.preprocessed_data)} samples")
    
    def _generate_cache_key(self, config: Dict[str, Any]) -> str:
        """Generate a unique cache key based on config parameters."""
        cache_params = {
            'dataset_name': config['dataset_name'],
            'dataset_size': config.get('dataset_size'),
            'text_encoder_model_name': config['text_encoder_model_name'],
            'audio_codec_model_name': config['audio_codec_model_name'],
            'max_prompt_length': config['max_prompt_length'],
            'max_audio_duration_sec': config.get('max_audio_duration_sec'),
            'dataset_path': config.get('dataset_path', '')
        }
        
        cache_str = str(sorted(cache_params.items()))
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _preprocess_dataset(self, config: Dict[str, Any], device) -> List[Dict[str, Any]]:
        """Preprocess the entire dataset and cache results."""
        # Initialize encoders
        text_embedder = TextEmbedder(config["text_encoder_model_name"], config["max_prompt_length"], device)
        audio_codec = AudioCodec(config["audio_codec_model_name"], config=config, device=device)
        
        # Load raw dataset
        dataset_size = config.get("dataset_size", None)
        d = load_dataset(
            config["dataset_name"], 
            split=self.split_name, 
            trust_remote_code=True
        )
        
        if dataset_size is not None:
            d = d.select(range(min(dataset_size, len(d))))

        raw_dataset = self.dict_to_records(d)
        
        preprocessed_data = []
        failed_samples = []
        
        print("Preprocessing samples...")
        for idx, sample in enumerate(tqdm(raw_dataset, desc="Processing")):
            try:
                rich_text = self.construct_rich_text(sample)
                text_embedding, text_attention_mask = self._get_text_embedding(rich_text, text_embedder)
                wav_file_path = config.get("dataset_path", "") + sample.get("location", "")
                audio_tokens, audio_attention_mask = self._get_audio_tokens(wav_file_path, audio_codec)
                
                if audio_tokens is None:
                    failed_samples.append(idx)
                    continue
                    
                preprocessed_sample = {
                    'text_embedding': text_embedding.cpu(),
                    'text_attention_mask': text_attention_mask.cpu(),
                    'audio_tokens': audio_tokens.cpu(),
                    'audio_attention_mask': audio_attention_mask.cpu(),
                    'sample_id': idx
                }
                preprocessed_data.append(preprocessed_sample)
                
            except Exception as e:
                print(f"Failed to process sample {idx}: {e}")
                failed_samples.append(idx)
                continue
        
        print(f"Successfully preprocessed {len(preprocessed_data)} samples")
        
        # Clean up encoders
        del text_embedder, audio_codec
        torch.cuda.empty_cache()
        
        return preprocessed_data
    
    def _save_cache(self):
        """Save preprocessed data to cache file."""
        print(f"Saving cache to {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.preprocessed_data, f)
    
    def _load_cache(self) -> List[Dict[str, Any]]:
        """Load preprocessed data from cache file."""
        with open(self.cache_file, 'rb') as f:
            return pickle.load(f)
    
    def dict_to_records(self, dataset_dict):
        """Convert HuggingFace dataset dict format to list of records."""
        if hasattr(dataset_dict, 'to_pandas'):
            return dataset_dict.to_pandas().to_dict('records')
        else:
            sample_key = next(iter(dataset_dict))
            n_samples = len(dataset_dict[sample_key])
            records = []
            for i in range(n_samples):
                record = {key: values[i] for key, values in dataset_dict.items()}
                records.append(record)
            return records
    
    def construct_rich_text(self, sample: Dict[str, Any]) -> str:
        """
        Construct rich text description from sample metadata.
        Combines main caption and other prompt fields.
        """
        text_parts = []
        
        main_caption = sample.get("main_caption")
        if main_caption and isinstance(main_caption, str) and main_caption.strip():
            text_parts.append(f"Description: {main_caption}")
        
        for field_name in ["prompt_ch", "prompt_bt", "prompt_bpm", "prompt_key"]:
            field_val = sample.get(field_name)
            if field_val:
                val_str = ""
                if isinstance(field_val, list) and field_val:
                    val_str = ', '.join(map(str, field_val))
                elif isinstance(field_val, str) and field_val.strip():
                    val_str = field_val
                
                if val_str:
                    clean_field_name = field_name.replace("prompt_", "").capitalize()
                    text_parts.append(f"{clean_field_name}: {val_str}")
        
        return ". ".join(text_parts) if text_parts else "An instrumental music piece."
    
    def _get_text_embedding(self, text: str, text_embedder) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get text embedding using the text encoder."""
        embeddings, attention_mask = text_embedder.encode([text])
        embeddings = embeddings.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        return embeddings, attention_mask
    
    def _get_audio_tokens(self, audio_path: str, audio_codec) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get audio tokens from audio file path."""
        try:
            tokens, attn_mask = audio_codec.encode_audio(audio_path, normalize=True)
            if tokens.dim() == 3 and tokens.shape[0] == 1:
                tokens = tokens.squeeze(0)
            return tokens, attn_mask
            
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            raise e
    
    def apply_cfg_dropout(self, text_embedding: torch.Tensor, text_attention_mask: torch.Tensor, 
                         audio_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply classifier-free guidance dropout during training."""
        cfg_mask = torch.zeros(2, dtype=torch.bool)
        
        if self.split_name == 'train' and torch.rand(1).item() < self.cfg_dropout_prob:
            text_embedding = torch.zeros_like(text_embedding)
            text_attention_mask = torch.zeros_like(text_attention_mask)
            cfg_mask[0] = True
        
        return text_embedding, text_attention_mask, audio_tokens, cfg_mask
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.preprocessed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the preprocessed dataset."""
        sample = self.preprocessed_data[idx]
        
        # Move tensors to device and apply CFG dropout
        text_embedding = sample['text_embedding'].to(self.device)
        text_attention_mask = sample['text_attention_mask'].to(self.device)
        audio_tokens = sample['audio_tokens'].to(self.device)
        audio_attention_mask = sample['audio_attention_mask'].to(self.device)
        
        # Apply CFG dropout
        text_embedding, text_attention_mask, audio_tokens, cfg_mask = self.apply_cfg_dropout(
            text_embedding, text_attention_mask, audio_tokens
        )
        
        return {
            'text_embedding': text_embedding,
            'text_attention_mask': text_attention_mask,
            'audio_tokens': audio_tokens,
            'audio_attention_mask': audio_attention_mask,
            'cfg_mask': cfg_mask,
        }


def optimized_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Optimized collate function for batching samples."""
    batch_size = len(batch)
    
    max_text_len = max(sample['text_embedding'].shape[0] for sample in batch)
    max_audio_len = max(sample['audio_tokens'].shape[-1] for sample in batch)
    n_codebooks = batch[0]['audio_tokens'].shape[0]
    embed_dim = batch[0]['text_embedding'].shape[-1]
    
    device = batch[0]['text_embedding'].device
    
    text_embeddings = torch.zeros(batch_size, max_text_len, embed_dim, device=device)
    text_attention_masks = torch.zeros(batch_size, max_text_len, dtype=torch.bool, device=device)
    audio_tokens = torch.zeros(batch_size, n_codebooks, max_audio_len, dtype=torch.long, device=device)
    audio_attention_masks = torch.zeros(batch_size, max_audio_len, dtype=torch.bool, device=device)
    cfg_masks = torch.stack([sample['cfg_mask'] for sample in batch])
    
    for i, sample in enumerate(batch):
        text_embeddings[i] = sample['text_embedding']
        text_attention_masks[i] = sample['text_attention_mask']
        audio_tokens[i] = sample['audio_tokens']
        audio_attention_masks[i] = sample['audio_attention_mask']
    
    return {
        'text_embedding': text_embeddings,
        'text_attention_mask': text_attention_masks,
        'audio_tokens': audio_tokens,
        'audio_attention_mask': audio_attention_masks,
        'cfg_mask': cfg_masks
    }
