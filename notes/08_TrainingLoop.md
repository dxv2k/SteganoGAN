# Training Loop Guide

This guide explains the **training loop** for the deep learning-based steganography model. The loop alternates between training the **critic (discriminator)** and the **encoder-decoder network**, ensuring stable adversarial learning.

## Table of Contents

- [Overview](#overview)
- [Initialization](#initialization)
- [Training Process](#training-process)
  - [Critic Update](#critic-update)
  - [Encoder-Decoder Update](#encoder-decoder-update)
- [Logging and Evaluation](#logging-and-evaluation)
- [Demo Code](#demo-code)

---

## Overview

The training loop is responsible for:

- **Training the critic** to distinguish between real and encoded images.
- **Training the encoder-decoder** to embed and extract secret data.
- **Using a gradient penalty** to stabilize adversarial learning.

The process follows a **WGAN-GP (Wasserstein GAN with Gradient Penalty)** setup.

---

## Initialization

```python
if __name__ == '__main__':
    dataloader = DataLoader(cover_image_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = SteganoModel(data_depth=8, block_size=64)
    model = model.to(device)
    optimizer_enc_dec = optim.AdamW(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=0.0001)
    optimizer_critic = optim.AdamW(model.critic.parameters(), lr=0.0001)

    criterion_d = nn.CrossEntropyLoss()
    criterion_s = nn.MSELoss()
    criterion_r = lambda x: -torch.mean(x)

    lambda_r = 1.0
    lambda_gp = 10.0
    num_epochs = 20
```

- **Dataloader**: Loads cover images in batches.
- **Model Instantiation**: Creates the steganography model and moves it to the GPU/CPU.
- **Optimizers**:
  - `optimizer_enc_dec`: Trains the **encoder** and **decoder**.
  - `optimizer_critic`: Trains the **critic**.
- **Loss Functions**:
  - `criterion_d`: Measures how well secret data is recovered.
  - `criterion_s`: Ensures encoded images remain close to the cover image.
  - `criterion_r`: Regularizes the training by maximizing critic scores.
- **Hyperparameters**:
  - `lambda_r = 1.0`: Regularization weight.
  - `lambda_gp = 10.0`: Gradient penalty weight.
  - `num_epochs = 20`: Number of training epochs.

---

## Training Process

### Critic Update

```python
for epoch in range(num_epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    total_critic_loss = 0.0
    total_enc_dec_loss = 0.0

    for images in progress_bar:
        images = images.to(device)
        B, C, H, W = images.size()
        embedding_data = torch.bernoulli(torch.full((B, 8, H, W), 0.5)).to(device)

        optimizer_critic.zero_grad()
        real_image_score = model.critic(images, embedding_data)
        encoded_image, _, _ = model(images, embedding_data)
        encoded_image_score = model.critic(encoded_image, embedding_data)

        loss_c = -(torch.mean(real_image_score) - torch.mean(encoded_image_score))
        gp = gradient_penalty(model.critic, images, encoded_image)
        loss_c += lambda_gp * gp
        loss_c.backward()
        optimizer_critic.step()
        total_critic_loss += loss_c.item()
```

- **Step 1**: Load a batch of images.
- **Step 2**: Generate random **binary secret data** (`embedding_data`).
- **Step 3**: Compute critic scores:
  - `real_image_score`: Score for real images.
  - `encoded_image_score`: Score for encoded images.
- **Step 4**: Compute **WGAN loss** for the critic.
- **Step 5**: Compute **gradient penalty** for stability.
- **Step 6**: Update the critic using **backpropagation**.

---

### Encoder-Decoder Update

```python
        optimizer_enc_dec.zero_grad()
        encoded_image, decoded_data, critic_score = model(images, embedding_data)
        loss_d = criterion_d(decoded_data, embedding_data)
        loss_s = criterion_s(encoded_image, images)
        loss_r = criterion_r(critic_score)
        total_loss = loss_d + loss_s + lambda_r * loss_r
        total_loss.backward()
        optimizer_enc_dec.step()
        total_enc_dec_loss += total_loss.item()
        progress_bar.set_postfix(critic_loss=loss_c.item(), enc_dec_loss=total_loss.item())
```

- **Step 1**: Compute encoded image and decoded secret data.
- **Step 2**: Compute individual loss components:
  - `loss_d`: Measures accuracy of recovered secret data.
  - `loss_s`: Ensures encoded image remains similar to the cover image.
  - `loss_r`: Regularization term to fool the critic.
- **Step 3**: Compute **total loss** for the encoder-decoder.
- **Step 4**: Backpropagate and update the encoder-decoder network.

---

## Logging and Evaluation

```python
    avg_critic_loss = total_critic_loss / len(dataloader)
    avg_enc_dec_loss = total_enc_dec_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Critic Loss: {avg_critic_loss:.4f}, Avg Enc-Dec Loss: {avg_enc_dec_loss:.4f}")
```

- **Logs average loss per epoch**:
  - `avg_critic_loss`: Monitors how well the critic is learning.
  - `avg_enc_dec_loss`: Monitors the effectiveness of encoding and decoding.

---

## Demo Code

Below is a **simplified version** of the training loop:

```python
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Dummy dataset
dataset = torch.utils.data.TensorDataset(torch.rand(100, 3, 256, 256))
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Dummy model with encoder, decoder, and critic
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.encoder = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.decoder = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.critic = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, image, data):
        encoded_image = self.encoder(image)
        decoded_data = self.decoder(encoded_image)
        critic_score = self.critic(encoded_image)
        return encoded_image, decoded_data, critic_score

# Initialize model and optimizer
model = DummyModel()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(2):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")
    for images, in progress_bar:
        optimizer.zero_grad()
        encoded_image, decoded_data, critic_score = model(images, images)
        loss = criterion(decoded_data, images)
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())

print("Training complete!")
```

### Running the Demo

1. **Ensure Dependencies:**  
   Install required packages:

   ```bash
   pip install torch torchvision tqdm
   ```

2. **Run the Script:**  

   ```bash
   python demo_train.py
   ```

---

## Summary

- **Updates critic** using **WGAN loss + gradient penalty**.
- **Trains encoder-decoder** using **reconstruction loss + regularization**.
- **Logs loss values** to monitor training progress.
