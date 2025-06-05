"""
Contains all hyperparameters and settings for training and inference.
"""

CONFIG = {
    # Dataset configuration
    "dataset_name": "amaai-lab/MusicBench",
    "dataset_split_train": "train",  
    "dataset_split_eval": "test",   
    "dataset_path": 'MusicBench/datashare/',
    "dataset_size": 25000,
    
    # Text encoder configuration
    "text_encoder_model_name": "google/flan-t5-base", 
    "max_prompt_length": 256,
    "text_embedding_dim": 768,  # Output dimension of FLAN-T5
    
    # Audio codec configuration
    "target_audio_sample_rate": 24000,  
    "audio_codec_model_name": "facebook/encodec_24khz", 
    "max_audio_duration_sec": 15,     
    "max_audio_seq_len": 750,
    "num_codebooks": 2,
    
    # Training configuration
    "batch_size": 20,
    "cfg_dropout_rate": 0.2,
    
    # Model architecture
    "embedding_dim": 512,
    "vocab_size": 1024,  # Codebook size from EnCodec model
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 20,
    "d_ff": 2048,
    "dropout": 0.25,
    
    # Training hyperparameters
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "num_epochs": 100,
    "gradient_clip_norm": 1.0,
    
    # Classifier-Free Guidance
    "cfg_weight": 3.0,
    
    # Inference parameters
    "top_k": 0,
    "top_p": 0.9,
    "temperature": 1.0,
    
    # Checkpointing and caching
    "save_every": 2,
    "cache_dir": 'pre-processed-music-bench',
    "checkpoint_dir": 'checkpoints',
}
