# Deep Learning for Image Processing

A comprehensive deep learning project implementing multiple neural network architectures for image compression, denoising, generation, and understanding using TensorFlow/Keras.

## Overview

This project demonstrates the implementation and training of five different deep learning architectures on a cats and dogs dataset:
- **Autoencoder (AE)** - Image compression and reconstruction
- **Denoising Autoencoder (DAE)** - Noise removal from images
- **Variational Autoencoder (VAE)** - Improved image reconstruction with latent space learning
- **Generative Adversarial Network (GAN)** - High-quality image generation
- **Transformer** - Image captioning architecture

## Dataset

**Source**: Cats and Dogs Mini Dataset from Kaggle  
**Size**: 1000 images (500 cats, 500 dogs)  
**Split**: 80% training (800 images), 20% validation (200 images)  
**Image Size**: 128x128x3 (RGB)

## Architecture Details

### 1. Autoencoder (Compression)
- **Purpose**: Image compression and reconstruction
- **Architecture**:
  - Encoder: Conv2D(32) → MaxPool → Conv2D(16) → MaxPool
  - Decoder: Conv2DTranspose(16) → Conv2DTranspose(32) → Conv2D(3)
- **Loss**: Binary Crossentropy
- **Training**: 5 epochs

### 2. Denoising Autoencoder
- **Purpose**: Remove noise from corrupted images
- **Architecture**:
  - Encoder: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool
  - Decoder: Conv2DTranspose(64) → Conv2DTranspose(32) → Conv2D(3)
- **Noise Factor**: 0.2 (Gaussian noise)
- **Loss**: Mean Squared Error
- **Training**: 5 epochs

### 3. Variational Autoencoder (VAE)
- **Purpose**: Generative modeling with improved reconstruction
- **Latent Dimension**: 512
- **Architecture**:
  - Encoder: Conv2D(64,128,256) with BatchNorm → Dense(1024) → z_mean, z_log_var
  - Decoder: Dense → Reshape → Conv2DTranspose(256,128,64) with BatchNorm
- **Loss**: Reconstruction Loss + 0.01 × KL Divergence
- **Training**: 10 epochs

### 4. Generative Adversarial Network (GAN)
- **Purpose**: Generate realistic synthetic images
- **Latent Dimension**: 128
- **Generator**:
  - Dense(16×16×256) → Reshape → Conv2DTranspose layers with BatchNorm and LeakyReLU
  - Output: Tanh activation
- **Discriminator**:
  - Conv2D layers with LeakyReLU and Dropout(0.3)
  - Output: Sigmoid activation
- **Optimizers**: Adam (lr=0.0002, beta_1=0.5)
- **Loss**: Binary Crossentropy with label smoothing
- **Training**: 30 epochs

### 5. Transformer (Image Captioning)
- **Purpose**: Image understanding and caption generation
- **Architecture**:
  - Image Encoder: Conv2D layers → Reshape to sequence
  - Caption Decoder: Embedding → Multi-Head Attention → Dense
- **Vocab Size**: 5000
- **Model Dimension**: 512
- **Note**: Architecture demonstration only (not trained in this notebook)

## Requirements

```
tensorflow>=2.x
numpy
matplotlib
kagglehub
```

## Installation

```bash
pip install tensorflow numpy matplotlib kagglehub
```

## Usage

### Running the Complete Pipeline

The notebook executes the following workflow:

1. **Dataset Download and Preprocessing**
   - Downloads dataset from Kaggle
   - Normalizes images to [0, 1]
   - Creates training and validation splits

2. **Model Training**
   - Trains all models sequentially
   - Monitors loss metrics during training

3. **Visualization**
   - Generates comparison visualizations
   - Processes 10 real images through all models
   - Saves results to `pipeline_10_images_with_gan.png`

### Key Functions

```python
# Build models
ae = build_autoencoder()
dae = build_denoising_autoencoder()
vae = build_vae()
gan = build_gan()
caption_model = build_captioning_transformer()

# Train models
ae.fit(train_ds, epochs=5, validation_data=val_ds)
dae.fit(train_ds, epochs=5, validation_data=val_ds)
vae.fit(train_ds, epochs=10)
gan.fit(train_ds, epochs=30)

# Visualize results
visualize_system_results(ae, dae, vae, gan, val_ds)
process_real_images(ae, dae, vae, gan)
```

## Results

The notebook generates two types of visualizations:

### 1. System Results Visualization
- Original vs Compressed (Autoencoder)
- Noisy vs Denoised (Denoising Autoencoder)
- GAN Generated Images
- Transformer Caption Predictions

### 2. Pipeline Processing (10 Images)
A comprehensive grid showing 10 real images processed through all models:
- Column 1: Original Image
- Column 2: Compressed & Reconstructed (AE)
- Column 3: Noisy Image
- Column 4: Denoised Image (DAE)
- Column 5: VAE Reconstruction
- Column 6: GAN Generated Image

Output saved as: `pipeline_10_images_with_gan.png`

## Training Performance

### GPU Acceleration
- Utilizes GPU if available (tested on Google Colab with T4 GPU)
- Automatic fallback to CPU if GPU unavailable

### Training Times (approximate on T4 GPU)
- Autoencoder: ~5 seconds
- Denoising Autoencoder: ~6 seconds
- VAE: ~20 seconds
- GAN: ~70 seconds

## Model Characteristics

| Model | Purpose | Key Feature | Output Quality |
|-------|---------|-------------|----------------|
| Autoencoder | Compression | Fast, simple | Good reconstruction |
| Denoising AE | Noise Removal | Robust to noise | Clean images |
| VAE | Generation | Smooth latent space | Realistic variations |
| GAN | Generation | Adversarial training | High-quality synthesis |
| Transformer | Understanding | Attention mechanism | Semantic captions |

## Technical Highlights

- **Custom Training Loops**: Implemented for DAE, VAE, and GAN
- **Label Smoothing**: Used in GAN discriminator (0.9 for real, 0.1 for fake)
- **Batch Normalization**: Applied in VAE and GAN for stable training
- **Data Augmentation**: Gaussian noise injection for denoising task
- **Performance Optimization**: Dataset caching, shuffling, and prefetching

## Future Enhancements

- Train the Transformer model with actual captions
- Implement conditional GAN (cGAN) for controlled generation
- Add image segmentation models (U-Net, Mask R-CNN)
- Experiment with different architectures (ResNet, EfficientNet)
- Implement style transfer using VAE/GAN
- Add evaluation metrics (FID, IS, SSIM, PSNR)

## License

This project is for educational purposes.

## Acknowledgments

- Dataset: Cats and Dogs Mini Dataset by Aleema Parakatta (Kaggle)
- Framework: TensorFlow/Keras
- Platform: Google Colab with GPU support
