# DenseEncoder Class Guide

The `DenseEncoder` class is a deep convolutional neural network designed to embed secret data into a cover image. It follows a dense connection architecture where features from earlier layers are concatenated with later layers to improve the information embedding process.

## Table of Contents

- [Overview](#overview)
- [Initialization](#initialization)
- [Forward Pass](#forward-pass)
- [Network Architecture](#network-architecture)
  - [conv1](#conv1)
  - [conv2](#conv2)
  - [conv3](#conv3)
  - [conv4](#conv4)
- [Demo Code](#demo-code)

---

## Overview

The `DenseEncoder` is responsible for:

- Taking a **cover image** (3-channel RGB image) as input.
- Taking **secret data** (default 8-channel representation) as input.
- Embedding the secret data into the cover image using convolutional layers.
- Producing an **encoded image** that visually resembles the cover image but carries the hidden data.

This model is a key component in the deep learning-based steganography pipeline.

---

## Initialization

```python
def __init__(self, data_depth=8):
    super(DenseEncoder, self).__init__()
    self.data_depth = data_depth

    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.BatchNorm2d(32)
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(32 + data_depth, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.BatchNorm2d(32)
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.BatchNorm2d(32)
    )

    self.conv4 = nn.Sequential(
        nn.Conv2d(96, 3, kernel_size=3, stride=1, padding=1)
    )
```

- **Parameters:**
  - `data_depth`: Specifies the number of channels used for the secret data (default is 8).
- **Layers:**
  - Uses a sequence of convolutional layers with **LeakyReLU** activations and **BatchNorm**.
  - Employs **skip connections** where earlier feature maps are concatenated into deeper layers.

---

## Forward Pass

```python
def forward(self, image, data):
    x1 = self.conv1(image)
    x2_input = torch.cat([x1, data], dim=1)
    x2 = self.conv2(x2_input)
    x3_input = torch.cat([x1, x2], dim=1)
    x3 = self.conv3(x3_input)
    x4_input = torch.cat([x1, x2, x3], dim=1)
    encoded_image = self.conv4(x4_input)
    return encoded_image
```

- **Inputs:**
  - `image`: The original cover image (shape: `[batch, 3, H, W]`).
  - `data`: The secret data to be hidden (shape: `[batch, data_depth, H, W]`).
- **Process:**
  - The image passes through convolutional layers with skip connections.
  - The secret data is concatenated at intermediate layers for better information flow.
  - The output is an **encoded image** that resembles the input image but carries hidden information.
- **Output:**
  - `encoded_image`: The modified image with hidden data (shape: `[batch, 3, H, W]`).

---

## Network Architecture

### conv1

```python
self.conv1 = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.LeakyReLU(negative_slope=0.01, inplace=True),
    nn.BatchNorm2d(32)
)
```

- First convolutional layer:
  - Takes the input **cover image** (3 channels).
  - Uses 32 filters of size **3×3**.
  - Applies **LeakyReLU** activation and **BatchNorm**.

### conv2

```python
self.conv2 = nn.Sequential(
    nn.Conv2d(32 + data_depth, 32, kernel_size=3, stride=1, padding=1),
    nn.LeakyReLU(negative_slope=0.01, inplace=True),
    nn.BatchNorm2d(32)
)
```

- Takes the output of `conv1` and concatenates it with the secret data.
- Processes the **(32 + data_depth)-channel** input.
- Uses another **3×3 convolution**, **LeakyReLU**, and **BatchNorm**.

### conv3

```python
self.conv3 = nn.Sequential(
    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
    nn.LeakyReLU(negative_slope=0.01, inplace=True),
    nn.BatchNorm2d(32)
)
```

- Concatenates the outputs of `conv1` and `conv2` (total **64 channels**).
- Uses another **3×3 convolution**, **LeakyReLU**, and **BatchNorm**.

### conv4

```python
self.conv4 = nn.Sequential(
    nn.Conv2d(96, 3, kernel_size=3, stride=1, padding=1)
)
```

- Final layer: Takes all previous feature maps (total **96 channels**).
- Uses a final **3×3 convolution** to generate a **3-channel output image**.
- Does **not** use activation or batch normalization here to maintain color similarity.

---

## Demo Code

Below is an example demonstrating how to use the `DenseEncoder` class:

```python
import torch
import torch.nn as nn

# Define an instance of the DenseEncoder
encoder = DenseEncoder(data_depth=8)

# Dummy input: Cover image (batch_size=1, 3 channels, 256x256)
cover_image = torch.rand(1, 3, 256, 256)

# Dummy input: Secret data (batch_size=1, 8 channels, 256x256)
secret_data = torch.rand(1, 8, 256, 256)

# Forward pass: Encode the secret data into the cover image
encoded_image = encoder(cover_image, secret_data)

# Print the output shape
print("Encoded Image Shape:", encoded_image.shape)
```

### Expected Output

```
Encoded Image Shape: torch.Size([1, 3, 256, 256])
```

### Running the Demo

1. **Ensure Dependencies:**  
   Install required packages if you haven't already:

   ```bash
   pip install torch torchvision
   ```

2. **Save the Code:**  
   Place the `DenseEncoder` class and demo code in a Python script (e.g., `demo_dense_encoder.py`).
3. **Run the Script:**  

   ```bash
   python demo_dense_encoder.py
   ```

You should see an output confirming that the encoded image has the expected shape.

---
