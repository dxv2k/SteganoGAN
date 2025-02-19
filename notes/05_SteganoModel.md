# SteganoModel Class Guide

The `SteganoModel` class is a deep learning-based steganography model that integrates an **encoder**, **decoder**, and **evaluator**. It embeds secret data into images, extracts it back, and evaluates the quality of the steganography process.

## Table of Contents

- [Overview](#overview)
- [Initialization](#initialization)
- [Forward Pass](#forward-pass)
- [Network Components](#network-components)
  - [Encoder](#encoder)
  - [Decoder](#decoder)
  - [Critic (Evaluator)](#critic-evaluator)
- [Demo Code](#demo-code)

---

## Overview

The `SteganoModel` is responsible for:

- **Encoding:** Hiding secret data within an image using a convolutional encoder.
- **Decoding:** Extracting the secret data from the modified image.
- **Evaluating:** Assessing the quality of the steganographic process using a critic network.

The model processes images in **blocks** of a given size (`block_size`) to ensure better embedding performance.

---

## Initialization

```python
def __init__(self, data_depth=8, block_size=64):
    super(SteganoModel, self).__init__()
    self.encoder = DenseEncoder(data_depth=data_depth)
    self.decoder = DenseDecoder(data_depth=data_depth)
    self.critic = Evaluate()
    self.block_size = block_size
```

- **Parameters:**
  - `data_depth`: Number of channels in the secret data (default is 8).
  - `block_size`: Size of the blocks into which the image is divided for processing (default is 64×64 pixels).
- **Components:**
  - Uses a `DenseEncoder` to embed data into images.
  - Uses a `DenseDecoder` to extract data from images.
  - Uses an `Evaluate` model as a **critic** to score the quality of the process.

---

## Forward Pass

```python
def forward(self, image, data):
    B, C, H, W = image.size()
    encoded_image = torch.zeros_like(image)
    decoded_data = torch.zeros_like(data)

    for i in range(0, H, self.block_size):
        for j in range(0, W, self.block_size):
            image_block = image[:, :, i:i+self.block_size, j:j+self.block_size]
            data_block = data[:, :, i:i+self.block_size, j:j+self.block_size]

            encoded_block = self.encoder(image_block, data_block)
            decoded_block = self.decoder(encoded_block)

            encoded_image[:, :, i:i+self.block_size, j:j+self.block_size] = encoded_block
            decoded_data[:, :, i:i+self.block_size, j:j+self.block_size] = decoded_block

    evaluation_score = self.critic(encoded_image, decoded_data)
    return encoded_image, decoded_data, evaluation_score
```

- **Inputs:**
  - `image`: Cover image to hide secret data (shape: `[batch, 3, H, W]`).
  - `data`: Secret data to be hidden (shape: `[batch, data_depth, H, W]`).
- **Process:**
  - The image and data are divided into **blocks**.
  - Each block is processed separately using:
    - `DenseEncoder` → Produces an **encoded image block**.
    - `DenseDecoder` → Extracts the **decoded data block**.
  - Blocks are reassembled into full images.
  - The **Evaluate** network computes a final quality score.
- **Outputs:**
  - `encoded_image`: The modified image containing the hidden data.
  - `decoded_data`: The extracted secret data.
  - `evaluation_score`: A score assessing the quality of the encoding/decoding.

---

## Network Components

### Encoder

```python
self.encoder = DenseEncoder(data_depth=data_depth)
```

- Encodes secret data into the cover image.
- Uses multiple convolutional layers with skip connections.
- Outputs an **encoded image** that resembles the cover image.

### Decoder

```python
self.decoder = DenseDecoder(data_depth=data_depth)
```

- Extracts hidden data from the encoded image.
- Uses convolutional layers to reconstruct the secret data.
- Outputs **decoded data** that should match the original input.

### Critic (Evaluator)

```python
self.critic = Evaluate()
```

- Assesses how well the steganographic process works.
- Uses convolutional layers to score the **encoded image** and **decoded data**.
- Produces a **single evaluation score**.

---

## Demo Code

Below is an example demonstrating how to use the `SteganoModel` class:

```python
import torch
import torch.nn as nn

# Define an instance of the SteganoModel
stegano_model = SteganoModel(data_depth=8, block_size=64)

# Dummy input: Cover image (batch_size=1, 3 channels, 256x256)
cover_image = torch.rand(1, 3, 256, 256)

# Dummy input: Secret data (batch_size=1, 8 channels, 256x256)
secret_data = torch.rand(1, 8, 256, 256)

# Forward pass: Encode, decode, and evaluate
encoded_image, decoded_data, score = stegano_model(cover_image, secret_data)

# Print output shapes
print("Encoded Image Shape:", encoded_image.shape)
print("Decoded Data Shape:", decoded_data.shape)
print("Evaluation Score Shape:", score.shape)
```

### Expected Output

```
Encoded Image Shape: torch.Size([1, 3, 256, 256])
Decoded Data Shape: torch.Size([1, 8, 256, 256])
Evaluation Score Shape: torch.Size([1, 1])
```

### Running the Demo

1. **Ensure Dependencies:**  
   Install required packages if you haven't already:

   ```bash
   pip install torch torchvision
   ```

2. **Save the Code:**  
   Place the `SteganoModel` class and demo code in a Python script (e.g., `demo_stegano_model.py`).
3. **Run the Script:**  

   ```bash
   python demo_stegano_model.py
   ```

You should see an output confirming that the encoding, decoding, and evaluation are working correctly.

---
