# Conditional Music Generation with Transformer Decoders and Classifier-Free Guidance

This project implements a conditional music generation system using transformer-based architecture with classifier-free guidance (CFG), leveraging the MusicBench dataset for training and evaluation.

## Overview

The project encompasses a complete pipeline for text-to-music generation, including data preprocessing, model training, and inference with controllable generation capabilities.

### Key Features

- **Dataset**: Utilizes the MusicBench dataset containing audio files with rich textual descriptions
- **Text Processing**: Employs FLAN-T5 (Google) for text embedding generation
- **Audio Processing**: Uses EnCodec (Facebook) for audio tokenization and reconstruction  
- **Model**: Custom Transformer decoder architecture optimized for music generation
- **Training**: Autoregressive training with Classifier-Free Guidance integration
- **Inference**: CFG-enabled generation allowing controllable music synthesis

### Approach Overview

1. **Dataset Processing**: The MusicBench dataset provides paired music audio samples with detailed textual descriptions including chords, beats, tempo, and BPM information.

2. **Text Embedding**: Rich text descriptions are constructed by combining multiple metadata fields (main_caption, prompt_ch, prompt_bt, prompt_bpm, prompt_key) and encoded using FLAN-T5-base model.

3. **Audio Tokenization**: Audio waveforms are converted into discrete token sequences using EnCodec, producing 2 codebooks per audio frame for comprehensive representation.

4. **Model Architecture**: A transformer decoder with cross-attention mechanisms enables conditioning on text while maintaining autoregressive audio generation capabilities.

5. **Classifier-Free Guidance**: During training, text conditions are randomly dropped to enable both conditional and unconditional generation. During inference, outputs from both modes are interpolated for controllable generation.


## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd music-generation-with-transformers
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the dataset path in `config.py` or use the default HuggingFace dataset.

## Usage

### Training

To train the model from scratch:

```bash
python main.py --mode train --config config.py
```

Optional training parameters:
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Training batch size (default: 20)
- `--learning_rate`: Learning rate (default: 5e-4)
- `--dataset_size`: Subset of dataset to use (default: 25000)

### Inference

To generate music samples:

```bash
# Conditional generation
python main.py --mode inference --checkpoint checkpoints/checkpoint_epoch_50.pt --prompt "A cheerful piano melody"

# Unconditional generation  
python main.py --mode inference --checkpoint checkpoints/checkpoint_epoch_50.pt

# Batch generation with multiple prompts
python main.py --mode inference --checkpoint checkpoints/checkpoint_epoch_50.pt --batch_prompts prompts.txt
```

### Configuration

Key configuration parameters in `config.py`:

- **Dataset Settings**:
  - `dataset_name`: "amaai-lab/MusicBench"
  - `dataset_size`: 25000 samples
  - `target_audio_sample_rate`: 24000 Hz
  - `max_audio_duration_sec`: 15 seconds

- **Model Architecture**:
  - `d_model`: 512 (model dimension)
  - `n_heads`: 8 (attention heads)
  - `n_layers`: 20 (transformer layers)
  - `vocab_size`: 1024 (EnCodec codebook size)

- **Training Parameters**:
  - `learning_rate`: 5e-4
  - `num_epochs`: 100
  - `cfg_dropout_rate`: 0.2 (CFG dropout probability)
  - `cfg_weight`: 3.0 (CFG guidance strength)

## Model Architecture

### MusicTransformer

The core model is a transformer decoder with the following components:

1. **Audio Token Embeddings**: Separate embedding layers for each EnCodec codebook, combined via summation
2. **Text Projection**: Projects FLAN-T5 embeddings (768D) to model dimension (512D)
3. **Positional Encoding**: Sinusoidal positional embeddings for sequence modeling
4. **Transformer Layers**: Stack of decoder layers with self-attention and cross-attention
5. **Output Heads**: Separate prediction heads for each codebook

### Classifier-Free Guidance

CFG enables controllable generation by training the model to handle both conditional and unconditional scenarios:

- **Training**: Random text dropout creates unconditional training samples
- **Inference**: Interpolation between conditional and unconditional predictions
- **Control**: CFG weight parameter controls adherence to text conditioning

## Data Preprocessing

The preprocessing pipeline handles:

1. **Text Processing**: 
   - Combines multiple metadata fields into rich descriptions
   - Tokenizes and embeds using FLAN-T5
   - Applies padding and attention masking

2. **Audio Processing**:
   - Loads and resamples audio to 24kHz
   - Encodes using EnCodec to discrete tokens
   - Applies sequence padding and attention masking

3. **Caching**: Preprocessed samples are cached to disk for efficient retraining

## Training Strategy

- **Loss Function**: Cross-entropy loss with padding token masking
- **Optimization**: AdamW optimizer with cosine annealing schedule
- **Regularization**: Gradient clipping and dropout
- **Monitoring**: Per-codebook loss tracking and validation metrics

## Generated Output

The inference engine produces:

- **Audio Files**: WAV format at 24kHz sample rate
- **Visualizations**: Waveform plots and spectrograms
- **Organization**: Separate directories for conditional/unconditional samples

## Dataset Statistics

- **Total Samples**: 25,000 (50% of full MusicBench dataset)
- **Audio Duration**: Up to 15 seconds per sample
- **Text Fields**: Main captions plus chord, beat, BPM, and key prompts
- **Genres**: Diverse collection spanning multiple musical styles