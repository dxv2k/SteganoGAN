# DenseDecoder Class Guide

The `DenseDecoder` class is a deep convolutional neural network designed to extract hidden data from an encoded image. It follows a dense connection architecture where features from earlier layers are concatenated with later layers to improve the data extraction process.

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

The `DenseDecoder` is responsible for:

- Taking an **encoded image** as input.
- Extracting the **hidden data** from the image using convolutional layers.
- Producing an **output data tensor** that should closely resemble the original secret data.

This model is a key component in the deep learning-based steganography pipeline.

---

## Initialization

```python
def __init__(self, data_depth=8):
    super(DenseDecoder, self).__init__()
    self.data_depth = data_depth

    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.BatchNorm2d(32)
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.BatchNorm2d(32)
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.BatchNorm2d(32)
    )

    self.conv4 = nn.Sequential(
        nn.Conv2d(96, self.data_depth, kernel_size=3, stride=1, padding=1)
    )
```

- **Parameters:**
  - `data_depth`: Specifies the number of channels for the extracted secret data (default is 8).
- **Layers:**
  - Uses a sequence of convolutional layers with **LeakyReLU** activations and **BatchNorm**.
  - Employs **skip connections** where earlier feature maps are concatenated into deeper layers.

---

## Forward Pass

```python
def forward(self, encoded_image):
    x1 = self.conv1(encoded_image)
    x2 = self.conv2(x1)
    x3_input = torch.cat([x1, x2], dim=1)
    x3 = self.conv3(x3_input)
    x4_input = torch.cat([x1, x2, x3], dim=1)
    decoded_data = self.conv4(x4_input)
    return decoded_data
```

- **Inputs:**
  - `encoded_image`: The modified image containing hidden data (shape: `[batch, 3, H, W]`).
- **Process:**
  - The encoded image passes through convolutional layers with skip connections.
  - The final output is an **extracted data tensor** with `data_depth` channels.
- **Output:**
  - `decoded_data`: The recovered secret data (shape: `[batch, data_depth, H, W]`).

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
  - Takes the **encoded image** as input (3 channels).
  - Uses 32 filters of size **3×3**.
  - Applies **LeakyReLU** activation and **BatchNorm**.

### conv2

```python
self.conv2 = nn.Sequential(
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
    nn.LeakyReLU(negative_slope=0.01, inplace=True),
    nn.BatchNorm2d(32)
)
```

- Takes the output of `conv1` and applies another **3×3 convolution**, **LeakyReLU**, and **BatchNorm**.

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
    nn.Conv2d(96, self.data_depth, kernel_size=3, stride=1, padding=1)
)
```

- Final layer: Takes all previous feature maps (total **96 channels**).
- Uses a final **3×3 convolution** to generate an **output tensor with `data_depth` channels**.
- Does **not** use activation or batch normalization here.

---

## Demo Code

Below is an example demonstrating how to use the `DenseDecoder` class:

```python
import torch
import torch.nn as nn

# Define an instance of the DenseDecoder
decoder = DenseDecoder(data_depth=8)

# Dummy input: Encoded image (batch_size=1, 3 channels, 256x256)
encoded_image = torch.rand(1, 3, 256, 256)

# Forward pass: Decode the hidden data from the encoded image
decoded_data = decoder(encoded_image)

# Print the output shape
print("Decoded Data Shape:", decoded_data.shape)
```

### Expected Output

```
Decoded Data Shape: torch.Size([1, 8, 256, 256])
```

### Running the Demo

1. **Ensure Dependencies:**  
   Install required packages if you haven't already:

   ```bash
   pip install torch torchvision
   ```

2. **Save the Code:**  
   Place the `DenseDecoder` class and demo code in a Python script (e.g., `demo_dense_decoder.py`).
3. **Run the Script:**  

   ```bash
   python demo_dense_decoder.py
   ```

You should see an output confirming that the decoded data has the expected shape.

---
