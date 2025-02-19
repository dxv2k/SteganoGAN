# CoverImageDataset Class Guide

The `CoverImageDataset` class is a custom PyTorch dataset used to load cover images for the steganography model. It allows preprocessing and transformation of images before feeding them into the neural network.

## Table of Contents

- [Overview](#overview)
- [Initialization](#initialization)
- [Dataset Length](#dataset-length)
- [Loading Images](#loading-images)
- [Demo Code](#demo-code)

---

## Overview

The `CoverImageDataset` is responsible for:

- Loading **cover images** from a specified directory.
- Applying **transformations** such as resizing and normalization.
- Preparing images for training in the **steganography pipeline**.

This dataset is used to train the **encoder** to embed secret data into images.

---

## Initialization

```python
def __init__(self, image_dir, block_size=64, transform=None):
    self.image_dir = image_dir
    self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    self.block_size = block_size
    self.transform = transform
```

- **Parameters:**
  - `image_dir`: The directory containing cover images.
  - `block_size`: Size of blocks used for processing (default is `64×64`).
  - `transform`: Optional image transformation (e.g., resizing, normalization).
- **File Loading:**
  - Reads all **.jpg** files in the specified `image_dir`.

---

## Dataset Length

```python
def __len__(self):
    return len(self.image_filenames)
```

- **Returns:**  
  - The total number of images in the dataset.

---

## Loading Images

```python
def __getitem__(self, idx):
    img_name = os.path.join(self.image_dir, self.image_filenames[idx])
    image = Image.open(img_name).convert("RGB")
    
    if self.transform:
        image = self.transform(image)

    return image
```

- **Process:**
  1. Loads an image by its index.
  2. Converts it to **RGB** format.
  3. Applies **transformations** (if specified).
- **Returns:**  
  - The transformed image as a PyTorch tensor.

---

## Demo Code

Below is an example demonstrating how to use the `CoverImageDataset` class:

```python
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Define a transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset instance
dataset = CoverImageDataset(image_dir='path_to_images', transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load a batch of images
for images in dataloader:
    print("Batch Shape:", images.shape)
    break  # Only show one batch
```

### Expected Output

```
Batch Shape: torch.Size([8, 3, 256, 256])
```

### Running the Demo

1. **Ensure Dependencies:**  
   Install required packages if you haven't already:

   ```bash
   pip install torch torchvision pillow
   ```

2. **Modify the Path:**  
   Replace `'path_to_images'` with the actual path to your image directory.
3. **Run the Script:**  

   ```bash
   python demo_dataset.py
   ```

You should see a batch of images loaded and transformed into tensors.

---

## Summary

- **Loads cover images from a directory** (`image_dir`).
- **Applies optional transformations** for resizing and normalization.
- **Returns images as PyTorch tensors** for use in the steganography model.
