# Evaluate Class Guide

The `Evaluate` class is a neural network designed to assess the quality of the encoded steganographic images and the extracted data. It uses separate convolutional pathways to process both the encoded images and the decoded data and combines their outputs into a final evaluation score.

## Table of Contents

- [Overview](#overview)
- [Initialization](#initialization)
- [Forward Pass](#forward-pass)
- [Network Architecture](#network-architecture)
  - [Image Processing Layers](#image-processing-layers)
  - [Data Processing Layers](#data-processing-layers)
  - [Fully Connected Layers](#fully-connected-layers)
- [Demo Code](#demo-code)

---

## Overview

The `Evaluate` network is responsible for:

- **Analyzing the encoded images**: Determines how well the secret data is embedded into the cover image.
- **Evaluating the decoded data**: Assesses the accuracy of the extracted secret data.
- **Producing a final quality score**: Uses convolutional layers followed by fully connected layers to compute a single evaluation score.

This class functions as the **critic** in an adversarial training setup, ensuring that the encoded images look realistic and that the secret data is correctly extracted.

---

## Initialization

```python
def __init__(self):
    super(Evaluate, self).__init__()

    # Layers for processing encoded images (3 channels)
    self.image_layers = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.BatchNorm2d(32),

        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.BatchNorm2d(32),

        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.BatchNorm2d(32),

        nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
    )

    # Layers for processing decoded data (data_depth channels)
    self.data_layers = nn.Sequential(
        nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.BatchNorm2d(32),

        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.BatchNorm2d(32),

        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.BatchNorm2d(32),

        nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
    )

    self.pool = nn.AdaptiveAvgPool2d(1)

    self.fc = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 1)
    )
```

- **Parameters:**
  - No input parameters; fixed architecture.
- **Layers:**
  - Two separate convolutional pathways: One for the encoded images and one for the decoded data.
  - Uses **LeakyReLU** activation and **BatchNorm** for stability.
  - A final **fully connected (FC) network** aggregates the results into a single score.

---

## Forward Pass

```python
def forward(self, encoded_image, decoded_data):
    # Process the encoded image with image_layers
    encoded_image_score = self.pool(self.image_layers(encoded_image)).view(encoded_image.size(0), -1)

    # Process the decoded data with data_layers
    decoded_data_score = self.pool(self.data_layers(decoded_data)).view(decoded_data.size(0), -1)

    # Combine the scores from both evaluations
    combined_score = torch.cat([encoded_image_score, decoded_data_score], dim=1)
    final_score = self.fc(combined_score)

    return final_score
```

- **Inputs:**
  - `encoded_image`: The modified image containing hidden data (shape: `[batch, 3, H, W]`).
  - `decoded_data`: The extracted secret data (shape: `[batch, 8, H, W]`).
- **Process:**
  - The `encoded_image` is processed through `image_layers`.
  - The `decoded_data` is processed through `data_layers`.
  - The outputs are pooled and combined into a **single vector**.
  - The **fully connected (FC) network** produces a **final score**.
- **Output:**
  - `final_score`: A scalar value assessing the quality of steganography (shape: `[batch, 1]`).

---

## Network Architecture

### Image Processing Layers

```python
self.image_layers = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.LeakyReLU(negative_slope=0.01, inplace=True),
    nn.BatchNorm2d(32),

    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
    nn.LeakyReLU(negative_slope=0.01, inplace=True),
    nn.BatchNorm2d(32),

    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
    nn.LeakyReLU(negative_slope=0.01, inplace=True),
    nn.BatchNorm2d(32),

    nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
)
```

- Processes the **encoded image** (3 channels).
- Uses multiple **3×3 convolutional layers** with **LeakyReLU** and **BatchNorm**.
- Reduces output to **1 channel** at the end.

### Data Processing Layers

```python
self.data_layers = nn.Sequential(
    nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
    nn.LeakyReLU(negative_slope=0.01, inplace=True),
    nn.BatchNorm2d(32),

    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
    nn.LeakyReLU(negative_slope=0.01, inplace=True),
    nn.BatchNorm2d(32),

    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
    nn.LeakyReLU(negative_slope=0.01, inplace=True),
    nn.BatchNorm2d(32),

    nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
)
```

- Processes the **decoded data** (8 channels).
- Uses the same architecture as the image pathway.

### Fully Connected Layers

```python
self.fc = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 1)
)
```

- Combines the processed image and data features into a **final evaluation score**.

---

## Demo Code

Below is an example demonstrating how to use the `Evaluate` class:

```python
import torch
import torch.nn as nn

# Define an instance of the Evaluate class
critic = Evaluate()

# Dummy input: Encoded image (batch_size=1, 3 channels, 256x256)
encoded_image = torch.rand(1, 3, 256, 256)

# Dummy input: Decoded data (batch_size=1, 8 channels, 256x256)
decoded_data = torch.rand(1, 8, 256, 256)

# Forward pass: Compute the evaluation score
score = critic(encoded_image, decoded_data)

# Print the output shape
print("Evaluation Score Shape:", score.shape)
```

### Expected Output

```
Evaluation Score Shape: torch.Size([1, 1])
```

### Running the Demo

1. **Ensure Dependencies:**  
   Install required packages if you haven't already:

   ```bash
   pip install torch torchvision
   ```

2. **Run the Script:**  

   ```bash
   python demo_evaluate.py
   ```

---
